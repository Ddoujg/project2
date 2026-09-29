# config.py
import os

TRAIN_CWT = '../train_dataset_fmd_cwt_30.h5'
VAL_CWT = '../val_dataset_fmd_cwt_30.h5'
TEST_CWT = '../test_dataset_fmd_cwt_30.h5'

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
DEVICE = 'cuda'
NUM_WORKERS = 0
CLIP_GRAD = 1.0

NUM_CLASSES = 10
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

# EViT 参数
EVIT_RATIO = 0.7            # 每层保留 token 的比例
SPARSITY_STAGES = [3, 6, 9] # 在第 3、6、9 层后执行剪枝（0-indexed）

OUTPUT_DIR = 'exp_C7_evit_results3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42