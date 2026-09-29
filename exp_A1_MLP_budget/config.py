# config.py
import os

# ---------- 数据路径 ----------
TRAIN_CWT = '../train_dataset_fmd_cwt_30.h5'
VAL_CWT = '../val_dataset_fmd_cwt_30.h5'
TEST_CWT = '../test_dataset_fmd_cwt_30.h5'

TRAIN_PHYS = '../train_physical_features.h5'
VAL_PHYS = '../val_physical_features.h5'
TEST_PHYS = '../test_physical_features.h5'

# ---------- 训练超参数 ----------
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_BUDGET = 5e-4
WEIGHT_DECAY_BACKBONE = 0.05
WEIGHT_DECAY_BUDGET = 0.01
DEVICE = 'cuda'
NUM_WORKERS = 0
CLIP_GRAD = 1.0

# ---------- 模型参数 ----------
NUM_CLASSES = 9
NUM_PHYSICAL_FEATURES = 8   # 剔除了低区分度特征后
IMG_SIZE = (64, 512)
PATCH_SIZE = (8, 16)
IN_CHANS = 3
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
MLP_RATIO = 4.0
DROP_RATE = 0.1
ATTN_DROP_RATE = 0.1
DROP_PATH_RATE = 0.1

# 预算参数
B_MIN = 3
B_MAX = 25
BUDGET_LAMBDA = 0.005       # 总预算约束权重，可降低至 0.05
TV_LAMBDA = 0.005
SPARSE_LAMBDA = 0.005


# Gumbel-Sinkhorn 参数
SELECTOR_NUM_ITER = 25
SELECTOR_TAU = 0.5
SELECTOR_EPS = 1.0
SELECTOR_USE_GUMBEL = True

# 输出目录
OUTPUT_DIR = './exp_mlp_budget_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
MONITOR_INTERVAL = 5