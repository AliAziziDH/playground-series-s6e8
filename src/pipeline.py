import pandas as pd
import numpy as np
from .data_loader import DataProcessor
from .model_trainer import TripleEnsembleTrainer
from .optimizer import ThresholdOptimizer

def run_pipeline(train_path: str, test_path: str, output_path: str = "submission.csv"):
    print("📥 Loading Dataset...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    processor = DataProcessor()
    X = processor.fit_transform(train_df.drop(columns=['id', 'target']))
    y = train_df['target']
    X_test = processor.transform(test_df.drop(columns=['id'], errors='ignore'))

    print("🧠 Training 5-Fold Stratified Triple Ensemble (CatBoost + LGBM + XGBoost)...")
    trainer = TripleEnsembleTrainer()
    oof_probs = trainer.train_cv(X, y)

    print("⚙️ Optimizing Per-Class Threshold Multipliers via Nelder-Mead...")
    optimizer = ThresholdOptimizer()
    optimizer.fit(oof_probs, y)

    print("🎯 Predicting Test Set & Generating Final Submission...")
    test_probs = trainer.predict_proba(X_test)
    test_preds = optimizer.predict(test_probs)

    sub = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    sub.to_csv(output_path, index=False)
    print(f"✅ Pipeline Execution Completed! Submission saved to {output_path}")

if __name__ == "__main__":
    run_pipeline("data/train.csv", "data/test.csv")
