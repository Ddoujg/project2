# config.py
import os

# 原始数据集路径（无噪声）
ORIGINAL_TRAIN_CWT = '../train_dataset_fmd_cwt_30.h5'
ORIGINAL_VAL_CWT = '../val_dataset_fmd_cwt_30.h5'
ORIGINAL_TEST_CWT = '../test_dataset_fmd_cwt_30.h5'

ORIGINAL_TRAIN_PHYS = '../train_physical_features.h5'
ORIGINAL_VAL_PHYS = '../val_physical_features.h5'
ORIGINAL_TEST_PHYS = '../test_physical_features.h5'

# 加噪数据集路径（仅内圈加噪，其他不变）
NOISY_TRAIN_CWT = '../train_dataset_fmd_cwt_noisy_30.h5'
NOISY_VAL_CWT = '../val_dataset_fmd_cwt_noisy_30.h5'
NOISY_TEST_CWT = '../test_dataset_fmd_cwt_noisy_30.h5'

NOISY_TRAIN_PHYS = '../train_physical_features_noisy.h5'
NOISY_VAL_PHYS = '../val_physical_features_noisy.h5'
NOISY_TEST_PHYS = '../test_physical_features_noisy.h5'

# 目标故障类标签（内圈故障，根据您的数据确定）
TARGET_CLASS = 5

# 训练超参数
NUM_CLASSES = 9
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_BUDGET = 2e-4
WEIGHT_DECAY_BACKBONE = 0.05
WEIGHT_DECAY_BUDGET = 0.01
DEVICE = 'cuda'
NUM_WORKERS = 0
CLIP_GRAD = 1.0

# 模型参数（与完整模型相同）
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
B_MIN = 2
B_MAX = 30
BUDGET_LAMBDA = 0.01
TV_LAMBDA = 0.01
SPARSE_LAMBDA = 0.005
W_SPARSE_LAMBDA = 0.001
SELECTOR_NUM_ITER = 20
SELECTOR_TAU = 0.5
SELECTOR_EPS = 1.0
SELECTOR_USE_GUMBEL = True

# 输出目录
OUTPUT_ORIGINAL = './exp_original_results'
OUTPUT_NOISY = './exp_noisy_results'
os.makedirs(OUTPUT_ORIGINAL, exist_ok=True)
os.makedirs(OUTPUT_NOISY, exist_ok=True)

SEED = 42