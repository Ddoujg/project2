# config.py
import os

# ---------- 数据路径 ----------
# 原始 .mat 文件所在目录（请根据实际情况修改）
MAT_DIR = r'D:\1111paper\paper2\su_origin_dataset'
# 用于划分训练/测试集的元数据（与预处理代码保持一致）
FILE_NAMES = ['ball_20_0', 'ball_30_2', 'comb_20_0', 'comb_30_2', 'inner_20_0', 'inner_30_2', 'outer_20_0',
              'outer_30_2', 'health_20_0']
DATA_COLUMN = ['ball_20_0', 'ball_30_2', 'comb_20_0', 'comb_30_2', 'inner_20_0', 'inner_30_2', 'outer_20_0',
              'outer_30_2', 'health_20_0']
# 样本划分参数（与预处理代码完全一致）
NORMAL_TRAIN_SAMPLES = 300   # 正常样本训练集数目
FAULT_TRAIN_SAMPLES = 30     # 每类故障训练集数目
VAL_SAMPLES_PER_CLASS = 10    # 每类验证集数目（这里作为测试集）
NORMAL_VAL_SAMPLES = 100      # 正常类验证集数目（保持与训练集相同的10:1比例）
TEST_SAMPLES_PER_CLASS = 10   #每类测试集数目
NORMAL_TEST_SAMPLES = 100
SEGMENT_LENGTH = 256
RANDOM_SEED = 42

# ---------- 训练超参数 ----------
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda'   # 或 'cpu'
NUM_WORKERS = 0

# ---------- 模型参数 ----------
NUM_CLASSES = 9   # 根据实际类别数修改
INPUT_CHANNELS = 1   # 原始信号单通道
SIGNAL_LEN = SEGMENT_LENGTH

# ---------- 输出目录 ----------
OUTPUT_DIR = './exp_C1_1DCNN_raw_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = RANDOM_SEED