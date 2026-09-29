# run_validation.py
import os
import json
import numpy as np
from config import ROOT_OUTPUT_DIR, FAULT_SAMPLE_SIZES, DEVICE
from train_eval import train_and_evaluate

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def main():
    results = []
    for n_fault in FAULT_SAMPLE_SIZES:
        print(f"\n=== Training with {n_fault} fault samples per class ===")
        out_dir = os.path.join(ROOT_OUTPUT_DIR, f'fault_samples_{n_fault}')
        res = train_and_evaluate(n_fault, out_dir, DEVICE)
        results.append(res)
    # 转换类型
    results_serializable = convert_to_serializable(results)
    with open(os.path.join(ROOT_OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print("Core idea validation completed.")

if __name__ == "__main__":
    main()