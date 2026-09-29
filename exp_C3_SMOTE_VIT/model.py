# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block as TimmBlock

from config import IMG_SIZE, PATCH_SIZE, IN_CHANS, EMBED_DIM, DEPTH, NUM_HEADS, MLP_RATIO, DROP_RATE, ATTN_DROP_RATE, DROP_PATH_RATE, NUM_CLASSES

# 自定义 Block，支持返回注意力
class CustomBlock(nn.Module):
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
        self.drop_path = nn.Identity()  # 简化，忽略 drop_path

    def forward(self, x, return_attention=False):
        if return_attention:
            # 使用位置参数调用 attn，避免关键字参数问题
            attn_out, attn_weights = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=True)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_weights
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class ViTWithAttention(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, qkv_bias=True, drop_rate=DROP_RATE,
                 attn_drop_rate=ATTN_DROP_RATE, drop_path_rate=DROP_PATH_RATE):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            CustomBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
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

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        attentions = []
        for blk in self.blocks:
            if return_attention:
                x, attn = blk(x, return_attention=True)
                attentions.append(attn)
            else:
                x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        if return_attention:
            # 取最后一层的注意力（所有头平均），并提取 CLS token 对 patch 的注意力
            last_attn = attentions[-1]  # [B, N+1, N+1]
            cls_attn = last_attn[:, 0, 1:]          # [B, N]
            # 重塑为特征图大小
            attn_map = cls_attn.reshape(B, self.num_patches_h, self.num_patches_w)
            return cls_out, attn_map
        else:
            return cls_out

class Classifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.vit = ViTWithAttention()
        self.head = nn.Linear(self.vit.embed_dim, num_classes)

    def forward(self, x, return_attention=False):
        if return_attention:
            cls_feat, attn_map = self.vit(x, return_attention=True)
            logits = self.head(cls_feat)
            return logits, attn_map
        else:
            cls_feat = self.vit(x)
            logits = self.head(cls_feat)
            return logits