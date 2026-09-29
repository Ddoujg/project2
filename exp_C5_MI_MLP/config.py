# config.py
import os

TRAIN_PHYS = '../train_physical_features.h5'
VAL_PHYS = '../val_physical_features.h5'
TEST_PHYS = '../test_physical_features.h5'

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda'
NUM_WORKERS = 0

NUM_CLASSES = 9
INPUT_DIM = 4           # 原始物理特征维度（4个特征）
NUM_SELECTED_FEATURES = 3  # 互信息选择 top-k 特征，可调整

OUTPUT_DIR = './exp_C5_mi_mlp_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42