# config.py
import os

# ---------- 数据路径 ----------
TRAIN_CWT = '../train_dataset_fmd_cwt_30.h5'
VAL_CWT = '../val_dataset_fmd_cwt_30.h5'
TEST_CWT = '../test_dataset_fmd_cwt_30.h5'
TRAIN_PHYS = '../train_physical_features.h5'
VAL_PHYS = '../val_physical_features.h5'
TEST_PHYS = '../test_physical_features.h5'

# ---------- 模型超参数（与完整模型一致）----------
B_MIN = 2
B_MAX = 25
BUDGET_LAMBDA = 0.01
TV_LAMBDA = 0.01
SPARSE_LAMBDA = 0.005
W_SPARSE_LAMBDA = 0.001

# ---------- 训练超参数 ----------
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_BUDGET = 2e-4
WEIGHT_DECAY_BACKBONE = 0.05
WEIGHT_DECAY_BUDGET = 0.01
DEVICE = 'cuda'
NUM_WORKERS = 0
CLIP_GRAD = 1.0

# ---------- 模型参数 ----------
NUM_CLASSES = 9
NUM_PHYSICAL_FEATURES = 12
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
SELECTOR_NUM_ITER = 20
SELECTOR_TAU = 0.5
SELECTOR_EPS = 1.0
SELECTOR_USE_GUMBEL = True

# ---------- 实验参数 ----------
# 待测试的故障样本数目（每类）
FAULT_SAMPLE_SIZES = [5, 10, 15, 20, 25]
# 固定正常样本数
NORMAL_SAMPLE_SIZE = 300

# ---------- 输出根目录 ----------
ROOT_OUTPUT_DIR = './results'
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

SEED = 42