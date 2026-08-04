import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from .config import N_SPLITS, RANDOM_STATE, MODEL_PARAMS

class TripleEnsembleTrainer:
    def __init__(self):
        self.models_cat = []
        self.models_lgb = []
        self.models_xgb = []

    def train_cv(self, X: pd.DataFrame, y: pd.Series):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        oof_preds = np.zeros((len(X), len(np.unique(y))))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            # Model 1: CatBoost (50%)
            cat = CatBoostClassifier(**MODEL_PARAMS["catboost"])
            cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
            self.models_cat.append(cat)

            # Model 2: LightGBM (30%)
            lgb = LGBMClassifier(**MODEL_PARAMS["lgbm"])
            lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            self.models_lgb.append(lgb)

            # Model 3: XGBoost (20%)
            xgb = XGBClassifier(**MODEL_PARAMS["xgboost"])
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            self.models_xgb.append(xgb)

            val_cat_prob = cat.predict_proba(X_val)
            val_lgb_prob = lgb.predict_proba(X_val)
            val_xgb_prob = xgb.predict_proba(X_val)

            oof_preds[val_idx] = 0.50 * val_cat_prob + 0.30 * val_lgb_prob + 0.20 * val_xgb_prob

        return oof_preds

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        test_preds = np.zeros((len(X_test), 2))
        for cat, lgb, xgb in zip(self.models_cat, self.models_lgb, self.models_xgb):
            p_cat = cat.predict_proba(X_test)
            p_lgb = lgb.predict_proba(X_test)
            p_xgb = xgb.predict_proba(X_test)
            test_preds += (0.50 * p_cat + 0.30 * p_lgb + 0.20 * p_xgb) / N_SPLITS
        return test_preds
