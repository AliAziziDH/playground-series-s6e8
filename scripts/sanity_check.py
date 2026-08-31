import pandas as pd
import numpy as np
import sys

def run_sanity_checks(file_path):
    print(f"Running sanity checks on {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"FAILED: Could not read file. {e}")
        sys.exit(1)

    # 1. Shape Integrity
    if df.shape != (296302, 2):
        print(f"FAILED: Shape Integrity. Expected (296302, 2), got {df.shape}")
        sys.exit(1)
    print("✅ Shape Integrity: Passed")

    # 2. Header Integrity
    expected_cols = ['id', 'addicted_label']
    if list(df.columns) != expected_cols:
        print(f"FAILED: Header Integrity. Expected {expected_cols}, got {list(df.columns)}")
        sys.exit(1)
    print("✅ Header Integrity: Passed")

    # 3. Null / NaN / Inf Check
    nulls = df.isnull().sum().sum()
    infs = np.isinf(df['addicted_label']).sum()
    if nulls > 0 or infs > 0:
        print(f"FAILED: Null/Inf Check. Found {nulls} nulls and {infs} infs.")
        sys.exit(1)
    print("✅ Null / NaN / Inf Check: Passed")

    # 4. Probability Bounds
    min_pred = df['addicted_label'].min()
    max_pred = df['addicted_label'].max()
    if min_pred < 0.0 or max_pred > 1.0:
        print(f"FAILED: Probability Bounds. Min: {min_pred}, Max: {max_pred}")
        sys.exit(1)
    print("✅ Probability Bounds: Passed")

    # 5. Prior Distribution Alignment
    mean_pred = df['addicted_label'].mean()
    if not (0.28 <= mean_pred <= 0.38):
        print(f"FAILED: Prior Distribution Alignment. Mean {mean_pred} is not in [0.28, 0.38]")
        sys.exit(1)
    print(f"✅ Prior Distribution Alignment: Passed (Mean = {mean_pred:.4f})")

    # 6. Deciles Inspection
    print("\nDeciles Inspection:")
    deciles = np.percentile(df['addicted_label'], np.arange(10, 100, 10))
    for i, dec in enumerate(deciles):
        print(f"  {(i+1)*10}th Percentile: {dec:.5f}")

    print("\n🎉 ALL SANITY CHECKS PASSED. Ready for submission.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sanity_check.py <path_to_submission_csv>")
        sys.exit(1)
    run_sanity_checks(sys.argv[1])
