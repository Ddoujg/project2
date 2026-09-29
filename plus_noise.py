import numpy as np
from scipy.io import loadmat, savemat


def add_noise_to_mat(input_mat_path, output_mat_path, snr_db, key=None):
    """
    对 .mat 文件中的信号添加高斯白噪声，保持其他变量不变。

    参数：
        input_mat_path: 输入 .mat 文件路径
        output_mat_path: 输出 .mat 文件路径
        snr_db: 信噪比 (dB)
        key: 需要加噪的变量名。若为 None，则自动选择第一个数值数组（通常是信号）。
    """
    # 加载 .mat 文件
    mat_data = loadmat(input_mat_path)

    # 确定需要加噪的变量
    if key is None:
        # 自动查找第一个数值数组（排除 '__header__', '__version__', '__globals__'）
        for k, v in mat_data.items():
            if k.startswith('__'):
                continue
            if isinstance(v, np.ndarray) and v.dtype.kind in 'fc':
                key = k
                break
        if key is None:
            raise ValueError("未找到可加噪的数值数组，请指定 key 参数。")

    signal = mat_data[key]
    if signal.ndim == 0:
        signal = signal.reshape(-1)

    # 计算信号功率
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise

    # 更新数据
    mat_data[key] = noisy_signal

    # 保存，保持原有结构
    savemat(output_mat_path, mat_data)
    print(f"已保存加噪后的文件至: {output_mat_path}")
    print(f"变量 '{key}' 已添加 SNR={snr_db} dB 的高斯白噪声")


# 示例用法
if __name__ == "__main__":
    # 对单个文件加噪
    add_noise_to_mat('su_origin_dataset/inner_20_0.mat', 'inner_20_0.mat', snr_db=-10, key='inner_20_0')