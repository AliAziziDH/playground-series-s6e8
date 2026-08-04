import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RANDOM_STATE = 42
N_SPLITS = 5

MODEL_PARAMS = {
    "catboost": {
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "random_seed": RANDOM_STATE,
        "verbose": 0
    },
    "lgbm": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "random_state": RANDOM_STATE,
        "verbose": -1
    },
    "xgboost": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "random_state": RANDOM_STATE,
        "eval_metric": "mlogloss"
    }
}
