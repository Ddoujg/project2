import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from config import (
    IMG_SIZE, PATCH_SIZE, IN_CHANS, EMBED_DIM, DEPTH, NUM_HEADS, MLP_RATIO,
    DROP_RATE, ATTN_DROP_RATE, DROP_PATH_RATE, NUM_CLASSES,
    B_MIN, B_MAX, TV_LAMBDA, SPARSE_LAMBDA
)
# ------------------------------
# ViT 骨干网络（输出特征图）
# ------------------------------
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
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
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        patch_tokens = x[:, 1:]
        feature_map = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, self.num_patches_h, self.num_patches_w)
        return feature_map, cls_out

# ------------------------------
# 解耦空间注意力
# ------------------------------
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

# ------------------------------
# Gumbel-Sinkhorn 选择器（无列约束）
# ------------------------------
class GumbelSinkhornSelector(nn.Module):
    def __init__(self, num_iter=50, tau=1.0, eps=0.1, use_gumbel=True):
        super().__init__()
        self.num_iter = num_iter
        self.tau = tau
        self.eps = eps
        self.use_gumbel = use_gumbel

    def forward(self, cost, budgets, features):
        N, Np, C = features.shape
        if self.training:
            if self.use_gumbel:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(cost) + 1e-8))
                cost = cost + self.tau * gumbel_noise
            u = torch.zeros(N, device=cost.device)
            scaled_cost = cost / self.eps
            for _ in range(self.num_iter):
                u = torch.log(budgets + 1e-8) - torch.logsumexp(-scaled_cost + u[:, None], dim=1)
            log_pi = -scaled_cost + u[:, None]
            pi = torch.exp(log_pi)
            agg_feat = (pi[:, :, None] * features).sum(dim=1) / budgets[:, None].clamp(min=1e-8)
            return agg_feat
        else:
            agg_feat = []
            for i in range(N):
                b = int(budgets[i].item())
                if b == 0:
                    agg_feat.append(torch.zeros(C, device=cost.device))
                else:
                    _, idx = torch.topk(-cost[i], k=b)
                    feat_i = features[i, idx, :].mean(dim=0)
                    agg_feat.append(feat_i)
            return torch.stack(agg_feat)

# ------------------------------
# 完整模型（固定预算）
# ------------------------------
class FixedBudgetModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, B_min=B_MIN, B_max=B_MAX,
                 tv_lambda=TV_LAMBDA, sparse_lambda=SPARSE_LAMBDA):
        super().__init__()
        self.B_min = B_min
        self.B_max = B_max
        self.tv_lambda = tv_lambda
        self.sparse_lambda = sparse_lambda
        self.fixed_budget = (B_min + B_max) / 2.0

        self.extractor = ViTWithSpatial()
        self.embed_dim = self.extractor.embed_dim
        self.num_patches_h = self.extractor.num_patches_h
        self.num_patches_w = self.extractor.num_patches_w

        self.attention = DecoupledSpatialAttention(in_channels=self.embed_dim, kernel_size=3)
        self.selector = GumbelSinkhornSelector(num_iter=20, tau=0.5, eps=1.0)
        self.pre_classifier_norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def temporal_tv_loss(self, S):
        diff = S[:, :, :, 1:] - S[:, :, :, :-1]
        return torch.abs(diff).mean()

    def frequency_sparse_loss(self, S):
        return torch.abs(S).mean()

    def forward(self, cwt_images, physical_features=None, labels=None):
        # physical_features 不再使用，但保留接口
        F_map, _ = self.extractor(cwt_images)
        N, C, H, W = F_map.shape
        S = self.attention(F_map)
        features = F_map.permute(0, 2, 3, 1).reshape(N, H*W, C)
        cost = -S.reshape(N, H*W)
        # 固定预算
        a = torch.full((N,), self.fixed_budget, device=cwt_images.device).float()

        z = self.selector(cost, a, features)
        z = self.pre_classifier_norm(z)
        logits = self.classifier(z)

        outputs = {'logits': logits, 'S': S, 'a': a}
        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            loss_tv = self.tv_lambda * self.temporal_tv_loss(S)
            loss_sparse = self.sparse_lambda * self.frequency_sparse_loss(S)
            loss = loss_cls + loss_tv + loss_sparse
            outputs['loss'] = loss
            outputs['loss_cls'] = loss_cls
            outputs['loss_tv'] = loss_tv
            outputs['loss_sparse'] = loss_sparse
        return outputs

    def inference(self, cwt_images, physical_features=None):
        self.eval()
        with torch.no_grad():
            F_map, _ = self.extractor(cwt_images)
            N, C, H, W = F_map.shape
            S = self.attention(F_map)
            features = F_map.permute(0, 2, 3, 1).reshape(N, H*W, C)
            cost = -S.reshape(N, H*W)
            a = torch.full((N,), self.fixed_budget, device=cwt_images.device).float()
            b = torch.round(a).long()
            agg_feat = []
            for i in range(N):
                bi = b[i].item()
                if bi == 0:
                    agg_feat.append(torch.zeros(C, device=cost.device))
                else:
                    _, idx = torch.topk(-cost[i], k=bi)
                    feat_i = features[i, idx, :].mean(dim=0)
                    agg_feat.append(feat_i)
            z = torch.stack(agg_feat)
            z = self.pre_classifier_norm(z)
            logits = self.classifier(z)
            preds = torch.argmax(logits, dim=1)
            return preds, S, b

if __name__ == "__main__":
    model = FixedBudgetModel()
    dummy_cwt = torch.randn(1, 3, 64, 512)
    dummy_phys = torch.randn(1, 12)
    out = model(dummy_cwt, dummy_phys)
    print(f"Output logits shape: {out['logits'].shape}")