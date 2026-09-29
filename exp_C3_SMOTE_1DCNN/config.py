# config.py
import os

# 原始数据路径（与 C1 相同）
MAT_DIR = r'D:\1111paper\paper2\su_origin_dataset'
FILE_NAMES = ['ball_20_0', 'ball_30_2', 'comb_20_0', 'comb_30_2', 'inner_20_0', 'inner_30_2', 'outer_20_0',
              'outer_30_2', 'health_20_0']
DATA_COLUMN = ['ball_20_0', 'ball_30_2', 'comb_20_0', 'comb_30_2', 'inner_20_0', 'inner_30_2', 'outer_20_0',
              'outer_30_2', 'health_20_0']
NORMAL_TRAIN_SAMPLES = 300
FAULT_TRAIN_SAMPLES = 30
VAL_SAMPLES_PER_CLASS = 10   # 验证集每类样本数，作为测试集
NORMAL_VAL_SAMPLES = 100      # 正常类验证集数目（保持与训练集相同的10:1比例）
TEST_SAMPLES_PER_CLASS = 10   #每类测试集数目
NORMAL_TEST_SAMPLES = 100
SEGMENT_LENGTH = 512
RANDOM_SEED = 42

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda'
NUM_WORKERS = 0
CLIP_GRAD = 1.0

NUM_CLASSES = 9
INPUT_CHANNELS = 1
SIGNAL_LEN = SEGMENT_LENGTH

OUTPUT_DIR = './exp_C3_smote_1dcnn_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = RANDOM_SEED