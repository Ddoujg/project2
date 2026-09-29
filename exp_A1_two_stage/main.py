# main.py
import subprocess
import sys
import os
from config import OUTPUT_DIR

def run_script(script_name):
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        sys.exit(1)

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_script("train_stage1.py")
    run_script("train_stage2.py")
    run_script("test.py")        # 使用 stage2 的最佳模型进行测试
    run_script("visualize.py")
    print("\nTwo-stage training completed.")