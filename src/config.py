import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RANDOM_STATE = 42
N_SPLITS = 10

MODEL_PARAMS = {
    "lgbm": {
        "num_leaves": 127,
        "min_child_samples": 150,
        "reg_lambda": 10.0,
        "learning_rate": 0.03,
        "random_state": RANDOM_STATE,
        "verbose": -1,
        "n_estimators": 1000
    },
    "xgboost": {
        "max_depth": 6,
        "reg_lambda": 5.0,
        "colsample_bytree": 0.7,
        "learning_rate": 0.03,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "n_estimators": 1000
    }
}
