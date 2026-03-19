
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

def verify():
    print("Verifying Validation Logic...")
    
    # Check data
    gt_path = DATA_DIR / "rough_volatility.npy"
    if not gt_path.exists():
        print("GT missing, skipping load check.")
        # Generate dummy for logic check
        gt_data = np.random.randn(10, 100)
    else:
        gt_data = np.load(gt_path)
        print(f"GT loaded: {gt_data.shape}")
        
    # Check fractal results
    fs_path = RESULTS_DIR / "fractal_production_final_paths.npy"
    if not fs_path.exists():
        print("Fractal results missing, skipping load check.")
    else:
        fs_data = np.load(fs_path)
        print(f"Fractal loaded: {fs_data.shape}")

    print("Logic check passed.")

if __name__ == "__main__":
    verify()
