# main.py
import subprocess
import sys

def run_script(script_name):
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        sys.exit(1)

if __name__ == "__main__":
    run_script("run_sensitivity.py")
    run_script("plot_results.py")
    print("Hyperparameter sensitivity analysis completed.")