import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from config import (
    IMG_SIZE, PATCH_SIZE, IN_CHANS, EMBED_DIM, DEPTH, NUM_HEADS, MLP_RATIO,
    DROP_RATE, ATTN_DROP_RATE, DROP_PATH_RATE, NUM_CLASSES,
    DYNAMIC_RATIO, SPARSITY_STAGES
)

# ------------------------------
# 动态剪枝预测模块（PredictorLG）
# ------------------------------
class PredictorLG(nn.Module):
    """轻量级预测模块，为每个 patch token 预测二值保留决策"""
    def __init__(self, embed_dim):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        # 拼接局部和全局特征，维度变为 2*embed_dim
        self.out_conv = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        """
        x: [B, N, C] patch tokens
        """
        # 局部特征
        local_x = self.in_conv(x)
        # 全局特征：对所有 token 平均池化
        global_x = local_x.mean(dim=1, keepdim=True).expand_as(local_x)
        # 拼接
        cat_x = torch.cat([local_x, global_x], dim=-1)
        score = self.out_conv(cat_x)
        return score


# ------------------------------
# 标准 ViT 骨干网络（支持动态剪枝）
# ------------------------------
class DynamicViT(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, qkv_bias=True, drop_rate=DROP_RATE,
                 attn_drop_rate=ATTN_DROP_RATE, drop_path_rate=DROP_PATH_RATE,
                 dynamic_ratio=DYNAMIC_RATIO, sparsity_stages=SPARSITY_STAGES):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.dynamic_ratio = dynamic_ratio
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
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])

        # 预测模块（仅在指定层后插入）
        self.predictors = nn.ModuleList([
            PredictorLG(embed_dim) for _ in range(len(sparsity_stages))
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
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.pos_embed
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)               # [B, 1+N, C]
        x = self.pos_drop(x)

        # 初始化保留决策（1=保留，0=丢弃），CLS token 始终保留
        keep_decision = torch.ones(B, 1 + self.num_patches, 1, device=x.device)
        keep_decision[:, 0, :] = 1.0

        pred_idx = 0
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # 在指定层后执行剪枝（不包括 CLS）
            if i in self.sparsity_stages and pred_idx < len(self.predictors):
                predictor = self.predictors[pred_idx]
                # 提取 patch tokens（不包含 CLS）
                patch_x = x[:, 1:, :]                     # [B, N, C]
                # 预测每个 patch 保留的概率
                score = predictor(patch_x)                # [B, N, 2]
                keep_prob = torch.exp(score[:, :, 0])     # [B, N]
                # 计算要保留的 patch 数量
                num_keep = max(1, int(self.num_patches * self.dynamic_ratio))
                # 选择 top-k patch
                _, indices = torch.topk(keep_prob, k=num_keep, dim=1)
                # 更新 patch 保留决策
                new_keep = torch.zeros_like(keep_decision[:, 1:, :])
                new_keep.scatter_(1, indices.unsqueeze(-1), 1.0)
                keep_decision = torch.cat([keep_decision[:, :1, :], new_keep], dim=1)
                pred_idx += 1

        # 最终输出：对保留的 token（含 CLS）进行加权平均
        keep_decision = keep_decision.float()
        x = (x * keep_decision).sum(dim=1) / (keep_decision.sum(dim=1) + 1e-8)
        x = self.norm(x)
        return x


class DynamicViTClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = DynamicViT()
        self.head = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits