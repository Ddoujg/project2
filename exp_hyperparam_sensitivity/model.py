# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from config import (
    IMG_SIZE, PATCH_SIZE, IN_CHANS, EMBED_DIM, DEPTH, NUM_HEADS, MLP_RATIO,
    DROP_RATE, ATTN_DROP_RATE, DROP_PATH_RATE, NUM_CLASSES, NUM_PHYSICAL_FEATURES,
    B_MIN, B_MAX, BUDGET_LAMBDA, TV_LAMBDA, SPARSE_LAMBDA, W_SPARSE_LAMBDA,
    SELECTOR_NUM_ITER, SELECTOR_TAU, SELECTOR_EPS, SELECTOR_USE_GUMBEL
)

# ------------------------------
# 1. ViT骨干网络（输出特征图）
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
# 2. 解耦空间注意力
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
# 3. 物理预算预测器
# ------------------------------
class PhysicalBudgetPredictor(nn.Module):
    def __init__(self, num_features, B_min=B_MIN, B_max=B_MAX):
        super().__init__()
        self.num_features = num_features
        self.B_min = B_min
        self.B_max = B_max
        self.weight_logits = nn.Parameter(torch.zeros(num_features))
        self.alpha_log = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, phys_feat):
        w = F.softplus(self.weight_logits)
        phi = (phys_feat * w).sum(dim=1)
        phi = torch.clamp(phi, -3.0, 3.0)
        alpha = torch.sigmoid(self.alpha_log) * 3.0
        r = torch.sigmoid(alpha * (phi - self.beta))
        a = self.B_min + (self.B_max - self.B_min) * r
        return a

    def get_intermediate(self, phys_feat):
        with torch.no_grad():
            w = F.softplus(self.weight_logits)
            phi = (phys_feat * w).sum(dim=1)
            phi = torch.clamp(phi, -3.0, 3.0)
            alpha = torch.sigmoid(self.alpha_log) * 3.0
            r = torch.sigmoid(alpha * (phi - self.beta))
            a = self.B_min + (self.B_max - self.B_min) * r
            return {
                'w': w.cpu().numpy(),
                'alpha': alpha.item(),
                'beta': self.beta.item(),
                'phi': phi.cpu().numpy(),
                'r': r.cpu().numpy(),
                'a': a.cpu().numpy()
            }

# ------------------------------
# 4. Gumbel-Sinkhorn选择器（无列约束）
# ------------------------------
class GumbelSinkhornSelector(nn.Module):
    def __init__(self, num_iter=SELECTOR_NUM_ITER, tau=SELECTOR_TAU, eps=SELECTOR_EPS, use_gumbel=SELECTOR_USE_GUMBEL):
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
# 5. 完整模型
# ------------------------------
class FMDGumbelSinkhornViTModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_physical_features=NUM_PHYSICAL_FEATURES,
                 B_min=B_MIN, B_max=B_MAX, budget_lambda=BUDGET_LAMBDA,
                 tv_lambda=TV_LAMBDA, sparse_lambda=SPARSE_LAMBDA, w_sparse_lambda=W_SPARSE_LAMBDA):
        super().__init__()
        self.B_min = B_min
        self.B_max = B_max
        self.budget_lambda = budget_lambda
        self.tv_lambda = tv_lambda
        self.sparse_lambda = sparse_lambda
        self.w_sparse_lambda = w_sparse_lambda

        self.extractor = ViTWithSpatial()
        self.embed_dim = self.extractor.embed_dim
        self.num_patches_h = self.extractor.num_patches_h
        self.num_patches_w = self.extractor.num_patches_w

        self.attention = DecoupledSpatialAttention(in_channels=self.embed_dim, kernel_size=3)
        self.budget_predictor = PhysicalBudgetPredictor(num_features=num_physical_features, B_min=B_min, B_max=B_max)
        self.selector = GumbelSinkhornSelector()
        self.pre_classifier_norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def temporal_tv_loss(self, S):
        # 沿时间轴（W维度）差分，适配冲击的时域连续性
        diff = S[:, :, :, 1:] - S[:, :, :, :-1]
        return torch.abs(diff).mean()

    def frequency_sparse_loss(self, S):
        # 频域L1稀疏正则
        return torch.abs(S).mean()

    def forward(self, cwt_images, physical_features, labels=None):
        F_map, _ = self.extractor(cwt_images)
        N, C, H, W = F_map.shape
        S = self.attention(F_map)
        features = F_map.permute(0, 2, 3, 1).reshape(N, H * W, C)
        cost = -S.reshape(N, H * W)
        a = self.budget_predictor(physical_features)

        # 固定批次总预算目标，基于单样本平均预算
        avg_budget = (self.B_min + self.B_max) / 2
        B_total_target = N * avg_budget

        z = self.selector(cost, a, features)
        z = self.pre_classifier_norm(z)
        logits = self.classifier(z)

        outputs = {'logits': logits, 'S': S, 'a': a}
        if labels is not None:
            # 分类损失
            loss_cls = F.cross_entropy(logits, labels)

            # 总预算约束（Huber损失）
            budget_error = a.sum() - B_total_target
            delta = B_total_target * 0.05
            huber_loss = torch.where(
                torch.abs(budget_error) <= delta,
                0.5 * budget_error ** 2,
                delta * (torch.abs(budget_error) - 0.5 * delta)
            )
            norm_factor = torch.tensor(abs(B_total_target), device=budget_error.device) + 1e-8
            loss_budget = self.budget_lambda * huber_loss / norm_factor

            # 正则项
            loss_tv = self.tv_lambda * self.temporal_tv_loss(S)
            loss_sparse = self.sparse_lambda * self.frequency_sparse_loss(S)

            # 物理特征权重L1稀疏正则
            w = F.softplus(self.budget_predictor.weight_logits)
            loss_w_sparse = self.w_sparse_lambda * torch.norm(w, p=1)

            loss = loss_cls + loss_budget + loss_tv + loss_sparse + loss_w_sparse

            outputs['loss'] = loss
            outputs['loss_cls'] = loss_cls
            outputs['loss_budget'] = loss_budget
            outputs['loss_tv'] = loss_tv
            outputs['loss_sparse'] = loss_sparse
            outputs['loss_w_sparse'] = loss_w_sparse
        return outputs

    def inference(self, cwt_images, physical_features):
        self.eval()
        with torch.no_grad():
            F_map, _ = self.extractor(cwt_images)
            N, C, H, W = F_map.shape
            S = self.attention(F_map)
            features = F_map.permute(0, 2, 3, 1).reshape(N, H*W, C)
            cost = -S.reshape(N, H*W)
            a = self.budget_predictor(physical_features)
            b = torch.round(a).long()
            B_target = int(N * ((self.B_min + self.B_max) / 2))
            delta = b.sum() - B_target
            if delta != 0:
                idx = torch.argsort(b, descending=True)
                for i in range(abs(delta).item()):
                    if delta > 0:
                        b[idx[i]] = max(self.B_min, b[idx[i]] - 1)
                    else:
                        b[idx[i]] = min(self.B_max, b[idx[i]] + 1)
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