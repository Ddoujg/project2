# config.py
import os

# ---------- 数据路径 ----------
TRAIN_CWT = '../train_dataset_fmd_cwt_30.h5'
VAL_CWT = '../val_dataset_fmd_cwt_30.h5'
TEST_CWT = '../test_dataset_fmd_cwt_30.h5'   # 请修改为实际测试集路径

TRAIN_PHYS = '../train_physical_features.h5'
VAL_PHYS = '../val_physical_features.h5'
TEST_PHYS = '../test_physical_features.h5'

# ---------- 训练超参数 ----------
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_BUDGET = 1e-4
WEIGHT_DECAY_BACKBONE = 0.05
WEIGHT_DECAY_BUDGET = 0.01
DEVICE = 'cuda'   # 或 'cpu'
NUM_WORKERS = 0
CLIP_GRAD = 1.0

# ---------- 模型参数 ----------
NUM_CLASSES = 10   # 根据实际类别修改
NUM_PHYSICAL_FEATURES = 4   # 根据实际物理特征维度修改

# ViT 参数
IMG_SIZE = (128, 512)
PATCH_SIZE = (8, 16)
IN_CHANS = 3
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
MLP_RATIO = 4.0
DROP_RATE = 0.1
ATTN_DROP_RATE = 0.05
DROP_PATH_RATE = 0.1

# 预算参数
B_MIN = 5
B_MAX = 25
BUDGET_LAMBDA = 0.01
TV_LAMBDA = 0.5          # 时域TV正则权重
SPARSE_LAMBDA = 0.5      # 频域稀疏正则权重
W_SPARSE_LAMBDA = 0.5    # 物理特征权重L1稀疏正则

# Gumbel-Sinkhorn 参数
SELECTOR_NUM_ITER = 20
SELECTOR_TAU = 0.5
SELECTOR_EPS = 1.0
SELECTOR_USE_GUMBEL = True

# 输出目录
OUTPUT_DIR = './exp_full_model_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 随机种子
SEED = 42

# 监控间隔
MONITOR_INTERVAL = 5