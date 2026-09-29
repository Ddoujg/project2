# config.py
import os

TRAIN_CWT = '../train_dataset_fmd_cwt_30.h5'
VAL_CWT = '../val_dataset_fmd_cwt_30.h5'
TEST_CWT = '../test_dataset_fmd_cwt_30.h5'
TRAIN_PHYS = '../train_physical_features.h5'
VAL_PHYS = '../val_physical_features.h5'
TEST_PHYS = '../test_physical_features.h5'

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
DEVICE = 'cuda'
NUM_WORKERS = 0
CLIP_GRAD = 1.0

NUM_CLASSES = 9
NUM_PHYSICAL_FEATURES = 4
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

# SMOTE 参数
SMOTE_K_NEIGHBORS = 5
SMOTE_SAMPLING_STRATEGY = 'auto'   # 使每类样本数相同

OUTPUT_DIR = './exp_C3_smote_vit_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42