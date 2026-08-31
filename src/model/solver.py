import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import scipy.stats
from sklearn.linear_model import LogisticRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import N_SPLITS, RANDOM_STATE, MODEL_PARAMS
import warnings
warnings.filterwarnings('ignore')

class ValueLevelTargetEncoder:
    def __init__(self, smoothing=10.0, cols=['gender', 'stress_level', 'academic_work_impact', 'daily_screen_time_hours_rounded', 'weekend_screen_time_rounded']):
        self.smoothing = smoothing
        self.cols = cols
        self.mappings_ = {}
        self.global_prior = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_encoded = X.copy()
        self.global_prior = y.mean()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for col in self.cols:
            if col not in X.columns:
                continue

            X_encoded[col + '_te'] = np.nan

            # Inner fold smoothing
            for train_idx, val_idx in skf.split(X, y):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val = X.iloc[val_idx]

                # compute stats
                stats = y_tr.groupby(X_tr[col]).agg(['sum', 'count'])
                smoothed = (stats['sum'] + self.smoothing * self.global_prior) / (stats['count'] + self.smoothing)

                # fill validation
                X_encoded.loc[X_val.index, col + '_te'] = X_val[col].map(smoothed).astype(np.float32)

            # Compute global mapping for test time
            stats = y.groupby(X[col]).agg(['sum', 'count'])
            smoothed = (stats['sum'] + self.smoothing * self.global_prior) / (stats['count'] + self.smoothing)
            self.mappings_[col] = smoothed.to_dict()

            # Fill missing levels with global prior
            X_encoded[col + '_te'] = X_encoded[col + '_te'].fillna(self.global_prior).astype(np.float32)

        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = X.copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X_encoded[col + '_te'] = X[col].map(self.mappings_[col]).fillna(self.global_prior).astype(np.float32)
        return X_encoded


class TripleEnsembleTrainer:
    def __init__(self):
        self.models_lgb = []
        self.models_xgb = []
        self.te = ValueLevelTargetEncoder()

    def train_cv(self, X: pd.DataFrame, y: pd.Series):
        # Target Encoding is applied via inner fold within train_cv or before?
        # Actually we need to apply target encoding without leakage, so we do it fold by fold?
        # ValueLevelTargetEncoder already does KFold internally for fit_transform! So we can apply it on the whole train set.
        # Wait, if we apply it on the whole train set using fit_transform, it generates leaked features?
        # The prompt says: inside an inner Stratified K-Fold (5 inner folds) during fit_transform().
        # This means we just call fit_transform on the whole X,y and it internally avoids leak for the main CV?
        # No, if the outer CV is 10-fold, we should fit_transform inside the 10-fold?
        # Actually, standard practice for Target Encoding with inner folds:
        # We can just fit_transform on the training set of the OUTER fold, and transform on the validation set of the OUTER fold.
        # Let's do that for maximum safety.

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        oof_preds = np.zeros((len(X), 2)) # probabilities

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()

            te = ValueLevelTargetEncoder()
            X_train_te = te.fit_transform(X_train, y_train)
            X_val_te = te.transform(X_val)

            # Drop the original categorical columns that were encoded
            cols_to_drop = [c for c in ['gender', 'stress_level', 'academic_work_impact', 'daily_screen_time_hours_rounded', 'weekend_screen_time_rounded'] if c in X_train.columns]
            X_train_te = X_train_te.drop(columns=cols_to_drop)
            X_val_te = X_val_te.drop(columns=cols_to_drop)

            # Keep numeric only for boosting? Or lightgbm can handle categoricals natively if we don't drop.
            # But the prompt says target encoding only to these. We will drop them to rely on TE.

            # Drop object/category types if any left to avoid model errors
            X_train_te = X_train_te.select_dtypes(exclude=['object', 'category'])
            X_val_te = X_val_te.select_dtypes(exclude=['object', 'category'])

            # Model 1: LightGBM
            # Since early_stopping_rounds is required for LightGBM, we need to pass callbacks or directly if older API
            # For newer LGBM, early_stopping_rounds goes in fit or via callbacks.
            lgb = LGBMClassifier(**MODEL_PARAMS["lgbm"])
            # Handling early stopping based on lightgbm version
            try:
                lgb.fit(X_train_te, y_train, eval_set=[(X_val_te, y_val)], callbacks=[early_stopping(50, verbose=False)])
            except:
                try:
                    lgb.fit(X_train_te, y_train, eval_set=[(X_val_te, y_val)], early_stopping_rounds=50, verbose=-1)
                except:
                    lgb.fit(X_train_te, y_train, eval_set=[(X_val_te, y_val)])

            self.models_lgb.append((lgb, te))

            # Model 2: XGBoost
            xgb = XGBClassifier(**MODEL_PARAMS["xgboost"])
            xgb.fit(X_train_te, y_train, eval_set=[(X_val_te, y_val)], verbose=False)
            self.models_xgb.append(xgb)

            val_lgb_prob = lgb.predict_proba(X_val_te)
            val_xgb_prob = xgb.predict_proba(X_val_te)

            # OOF probabilities from GBDT (we average them or just return both?
            # The prompt says: "blend only our newly trained GBDT models using Gauss-Rank Logistic Stacker"
            # So we should return them as separate columns in OOF to be blended!)
            # But wait, does it mean we blend lgb and xgb independently?
            # "Take out-of-fold probability vectors from our newly calibrated GBDT models alongside the existing public OOF library matrix."
            # So oof_preds should be shape (N, 2) where col 0 is lgb, col 1 is xgb.
            oof_preds[val_idx, 0] = val_lgb_prob[:, 1]
            oof_preds[val_idx, 1] = val_xgb_prob[:, 1]

        return oof_preds

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        test_preds = np.zeros((len(X_test), 2))
        for (lgb, te), xgb in zip(self.models_lgb, self.models_xgb):
            X_test_te = te.transform(X_test)
            cols_to_drop = [c for c in ['gender', 'stress_level', 'academic_work_impact', 'daily_screen_time_hours_rounded', 'weekend_screen_time_rounded'] if c in X_test_te.columns]
            X_test_te = X_test_te.drop(columns=cols_to_drop).select_dtypes(exclude=['object', 'category'])

            p_lgb = lgb.predict_proba(X_test_te)[:, 1]
            p_xgb = xgb.predict_proba(X_test_te)[:, 1]
            test_preds[:, 0] += p_lgb / N_SPLITS
            test_preds[:, 1] += p_xgb / N_SPLITS

        return test_preds

class OOFLibraryBlender:
    def __init__(self):
        self.meta_model = LogisticRegression(C=0.03, max_iter=1000, random_state=42, fit_intercept=True)

    def _gauss_rank(self, preds):
        r = scipy.stats.rankdata(preds) / len(preds)
        z = scipy.stats.norm.ppf(np.clip(r, 1e-5, 1.0 - 1e-5))
        return z

    def fit(self, our_oof: np.ndarray, y: pd.Series, oof_library: np.ndarray = None):
        if oof_library is not None:
            X_meta = np.column_stack([oof_library, our_oof])
        else:
            X_meta = our_oof

        # Apply Gauss-Rank transformation column-wise
        X_meta_transformed = np.zeros_like(X_meta)
        for i in range(X_meta.shape[1]):
            X_meta_transformed[:, i] = self._gauss_rank(X_meta[:, i])

        self.meta_model.fit(X_meta_transformed, y)

    def predict(self, our_test: np.ndarray, test_preds_library: np.ndarray = None):
        if test_preds_library is not None:
            X_test_meta = np.column_stack([test_preds_library, our_test])
        else:
            X_test_meta = our_test

        X_test_meta_transformed = np.zeros_like(X_test_meta)
        for i in range(X_test_meta.shape[1]):
            X_test_meta_transformed[:, i] = self._gauss_rank(X_test_meta[:, i])

        return self.meta_model.predict_proba(X_test_meta_transformed)[:, 1]
