# main.py
import subprocess
import sys

if __name__ == "__main__":
    print("Running strong features experiment (no noise)...")
    subprocess.run([sys.executable, 'train.py'])
    print("Running weak features experiment (adding noise to target class)...")
    # 注意：train.py 内部已经依次运行了强和弱两个实验，因此这里只需运行一次 train.py
    # 实际上 train.py 的 __main__ 中已经调用了两次 main()，所以直接运行即可。
    print("Generating comparison plot...")
    subprocess.run([sys.executable, 'visualize.py'])
    print("All experiments completed.")