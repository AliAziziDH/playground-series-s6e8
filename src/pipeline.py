import os
import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from .model.formulation import DataProcessor
from .model.solver import TripleEnsembleTrainer, OOFLibraryBlender

def run_pipeline(train_path: str, test_path: str, output_path: str = "outputs/submission_final_golden.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("📥 Loading Dataset...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    processor = DataProcessor()
    X = processor.fit_transform(train_df.drop(columns=['id', 'target']))
    y = train_df['target']
    X_test = processor.transform(test_df.drop(columns=['id'], errors='ignore'))

    print("🧠 Training 10-Fold Stratified Double Ensemble (LGBM + XGBoost)...")
    trainer = TripleEnsembleTrainer()
    oof_probs = trainer.train_cv(X, y) # shape (N, 2)

    print("⚙️ Blending with OOFLibraryBlender (Gauss-Rank Logistic Stacker)...")
    blender = OOFLibraryBlender()

    # Load 74-model OOF library if exists
    oof_library_path = "data/oof_library/train_oof.npy"
    test_library_path = "data/oof_library/test_preds.npy"

    oof_library, test_preds_library = None, None
    if os.path.exists(oof_library_path) and os.path.exists(test_library_path):
        print("📚 Loading 74-model OOF library...")
        oof_library = np.load(oof_library_path)
        test_preds_library = np.load(test_library_path)
    else:
        print("⚠️ OOF library not found. Falling back to blending only newly trained models.")

    blender.fit(oof_probs, y, oof_library)

    # Calculate and report OOF AUC
    oof_blended_preds = blender.predict(oof_probs, oof_library)
    oof_auc = roc_auc_score(y, oof_blended_preds)
    print(f"📊 Final 10-Fold OOF AUC: {oof_auc:.5f}")

    print("🎯 Predicting Test Set & Generating Final Submission...")
    test_probs = trainer.predict_proba(X_test)
    test_preds = blender.predict(test_probs, test_preds_library)

    sub = pd.DataFrame({"id": test_df["id"], "addicted_label": test_preds})
    sub.to_csv(output_path, index=False)
    print(f"✅ Pipeline Execution Completed! Submission saved to {output_path}")

    print("\n📈 Submission Distribution Statistics:")
    print(f"Min: {test_preds.min():.5f}")
    print(f"Max: {test_preds.max():.5f}")
    print(f"Mean: {test_preds.mean():.5f}")
    deciles = np.percentile(test_preds, np.arange(10, 100, 10))
    for i, dec in enumerate(deciles):
        print(f"Decile {(i+1)*10}%: {dec:.5f}")

if __name__ == "__main__":
    run_pipeline("data/train.csv", "data/test.csv")
    import subprocess
    print("\n🚀 Running final sanity checks...")
    result = subprocess.run(["python3", "scripts/sanity_check.py", "outputs/submission_final_golden.csv"])
    if result.returncode != 0:
        print("\n❌ Sanity checks failed! Halting submission.")
        exit(1)
    print("\n✅ Proceeding to Kaggle Submission Phase.")
