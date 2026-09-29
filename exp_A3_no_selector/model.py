# model.py
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from config import IMG_SIZE, PATCH_SIZE, IN_CHANS, EMBED_DIM, DEPTH, NUM_HEADS, MLP_RATIO, DROP_RATE, ATTN_DROP_RATE, DROP_PATH_RATE, NUM_CLASSES

# 解耦空间注意力（仅用于生成显著性图，但不用于特征选择）
class DecoupledSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.freq_conv = nn.Conv2d(in_channels, 1, kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.time_conv = nn.Conv2d(in_channels, 1, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        freq_att = self.freq_conv(x)
        time_att = self.time_conv(x)
        att = freq_att + time_att
        return self.sigmoid(att)

# ViT 骨干网络（输出特征图）
class ViTWithSpatial(nn.Module):
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
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
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
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # 重塑为特征图 [B, C, H_p, W_p]
        feature_map = x.transpose(1, 2).reshape(B, self.embed_dim, self.num_patches_h, self.num_patches_w)
        return feature_map

# 完整模型：ViT + 全局平均池化，无选择器
class GlobalAvgPoolModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.extractor = ViTWithSpatial()
        self.embed_dim = self.extractor.embed_dim
        self.attention = DecoupledSpatialAttention(in_channels=self.embed_dim, kernel_size=3)  # 仅用于可视化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x, return_attention=False):
        feat_map = self.extractor(x)          # [B, C, H, W]
        if return_attention:
            S = self.attention(feat_map)      # 显著性图（仅用于可视化）
        pooled = self.global_avg_pool(feat_map).squeeze(-1).squeeze(-1)  # [B, C]
        logits = self.classifier(pooled)
        if return_attention:
            return logits, S
        else:
            return logits