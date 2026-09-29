# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from config import (
    IMG_SIZE, PATCH_SIZE, IN_CHANS, EMBED_DIM, DEPTH, NUM_HEADS, MLP_RATIO,
    DROP_RATE, ATTN_DROP_RATE, DROP_PATH_RATE, NUM_CLASSES,
    EVIT_RATIO, SPARSITY_STAGES
)

# ------------------------------
# 自定义 Transformer 层，支持返回注意力权重
# ------------------------------
class TransformerBlockWithAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(proj_drop)
        )
        self.drop_path = nn.Identity()  # 简化，未使用 DropPath

    def forward(self, x, return_attention=False):
        # Self-attention
        x_norm = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=True)
            # attn_weights: [B, N, N]
        else:
            attn_out = self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.drop_path(attn_out)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn_weights
        else:
            return x

# ------------------------------
# EViT 骨干网络（基于注意力熵的 token 剪枝）
# ------------------------------
class EViT(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, qkv_bias=True, drop_rate=DROP_RATE,
                 attn_drop_rate=ATTN_DROP_RATE, drop_path_rate=DROP_PATH_RATE,
                 evit_ratio=EVIT_RATIO, sparsity_stages=SPARSITY_STAGES):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.evit_ratio = evit_ratio
        self.sparsity_stages = sparsity_stages

        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlockWithAttn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                     proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # [B, 1+N, C]
        x = self.pos_drop(x)

        # 保留决策（1=保留，0=丢弃），CLS token 始终保留
        keep = torch.ones(B, self.num_patches + 1, device=x.device)  # [B, N+1]

        # 遍历层
        for i, blk in enumerate(self.blocks):
            if i in self.sparsity_stages:
                # 获取注意力权重（CLS token 对所有 token 的注意力）
                x, attn_weights = blk(x, return_attention=True)  # attn_weights: [B, N+1, N+1]
                # 取 CLS token 对所有 token 的注意力（第0行，排除自身？通常取第0行）
                cls_attn = attn_weights[:, 0, 1:]   # [B, N]
                # 计算注意力熵
                entropy = -torch.sum(cls_attn * torch.log(cls_attn + 1e-8), dim=-1)  # [B]
                # 根据熵排序，保留熵最小的 token（注意力最集中的 token）
                # 实际 EViT 保留前 k 个 token，k = num_patches * evit_ratio
                num_keep = max(1, int(self.num_patches * self.evit_ratio))
                # 对每个样本独立选择保留的 token（基于 CLS 注意力权重）
                # 获取 top-k 索引（按注意力权重降序）
                _, indices = torch.topk(cls_attn, k=num_keep, dim=-1)  # [B, k]
                # 更新保留决策：生成新的 keep 掩码（保留 CLS token 和选中的 token）
                new_keep = torch.zeros_like(keep)
                new_keep[:, 0] = 1.0   # CLS token 始终保留
                # 将选中 token 的位置设为 1
                for b in range(B):
                    new_keep[b, indices[b] + 1] = 1.0   # +1 是因为第一个是 CLS
                keep = new_keep
                # 实际剪枝：保留 token 对应的特征，丢弃其余（这里不真正丢弃，而是用掩码加权）
                # 为了保持梯度，我们使用掩码加权（后续平均池化时使用）
                # 继续前向传播（不修改 x，因为后续层需要完整 token 序列）
                # 但 EViT 原文是直接丢弃 token，这里简化处理：在最后池化时根据 keep 加权平均
            else:
                x = blk(x, return_attention=False)

        # 最终池化：根据 keep 掩码对 token 特征进行加权平均（仅保留的 token 参与）
        keep = keep.unsqueeze(-1)   # [B, N+1, 1]
        x = (x * keep).sum(dim=1) / (keep.sum(dim=1) + 1e-8)   # [B, C]
        x = self.norm(x)
        return x

class EViTClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = EViT()
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits