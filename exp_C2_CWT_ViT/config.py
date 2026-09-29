# config.py
import os

# ---------- 数据路径 ----------
TRAIN_H5 = '../train_dataset_fmd_cwt_30.h5'
VAL_H5 = '../val_dataset_fmd_cwt_30.h5'
TEST_H5 = '../test_dataset_fmd_cwt_30.h5'   # 请修改为实际测试集路径

# ---------- 训练超参数 ----------
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
DEVICE = 'cuda'   # 或 'cpu'
NUM_WORKERS = 0

# ---------- ViT 参数 ----------
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

NUM_CLASSES = 10   # 根据实际类别修改

# ---------- 输出目录 ----------
OUTPUT_DIR = './exp_C2_CWT_ViT_results3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42