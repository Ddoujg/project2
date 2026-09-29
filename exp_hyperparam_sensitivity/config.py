# config.py
import os

# ---------- 数据路径（原始完整训练集、验证集、测试集）----------
TRAIN_CWT = 'train_dataset_fmd_cwt_30.h5'
VAL_CWT = 'val_dataset_fmd_cwt_30.h5'
TEST_CWT = 'test_dataset_fmd_cwt_30.h5'
TRAIN_PHYS = 'train_physical_features.h5'
VAL_PHYS = 'val_physical_features.h5'
TEST_PHYS = 'test_physical_features.h5'

# ---------- 默认超参数 ----------
DEFAULT_B_MIN = 2
DEFAULT_B_MAX = 25
DEFAULT_BUDGET_LAMBDA = 0.01
DEFAULT_TV_LAMBDA = 0.01
DEFAULT_SPARSE_LAMBDA = 0.005
DEFAULT_W_SPARSE_LAMBDA = 0.001

# 不均衡比例相关：原始训练集中正常样本数、故障样本总数（9类×30）
ORIGINAL_NORMAL_COUNT = 400
ORIGINAL_FAULT_COUNT = 270   # 9*30

# ---------- 训练超参数 ----------
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_BUDGET = 2e-4
WEIGHT_DECAY_BACKBONE = 0.05
WEIGHT_DECAY_BUDGET = 0.01
DEVICE = 'cuda'
NUM_WORKERS = 4
CLIP_GRAD = 1.0

# ---------- 模型参数 ----------
NUM_CLASSES = 10
NUM_PHYSICAL_FEATURES = 4
IMG_SIZE = (128, 512)
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

# ---------- 输出根目录 ----------
ROOT_OUTPUT_DIR = './exp_hyperparam_sensitivity/results'
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

SEED = 42