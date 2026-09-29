# main.py
from train import train_experiment
from config import (
    ORIGINAL_TRAIN_CWT, ORIGINAL_TRAIN_PHYS,
    ORIGINAL_VAL_CWT, ORIGINAL_VAL_PHYS,
    ORIGINAL_TEST_CWT, ORIGINAL_TEST_PHYS,
    NOISY_TRAIN_CWT, NOISY_TRAIN_PHYS,
    NOISY_VAL_CWT, NOISY_VAL_PHYS,
    NOISY_TEST_CWT, NOISY_TEST_PHYS,
    OUTPUT_ORIGINAL, OUTPUT_NOISY
)

if __name__ == "__main__":
    print("Training on original dataset...")
    train_experiment(
        ORIGINAL_TRAIN_CWT, ORIGINAL_TRAIN_PHYS,
        ORIGINAL_VAL_CWT, ORIGINAL_VAL_PHYS,
        ORIGINAL_TEST_CWT, ORIGINAL_TEST_PHYS,
        OUTPUT_ORIGINAL
    )
    print("Training on noisy dataset...")
    train_experiment(
        NOISY_TRAIN_CWT, NOISY_TRAIN_PHYS,
        NOISY_VAL_CWT, NOISY_VAL_PHYS,
        NOISY_TEST_CWT, NOISY_TEST_PHYS,
        OUTPUT_NOISY
    )
    print("Generating comparison plot...")
    import visualize
    visualize.plot_comparison()
    print("All experiments completed.")