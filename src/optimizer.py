import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

class ThresholdOptimizer:
    def __init__(self):
        self.best_multipliers = None

    def _loss_func(self, weights, oof_probs, y_true):
        w_probs = oof_probs * weights
        preds = np.argmax(w_probs, axis=1)
        score = f1_score(y_true, preds, average='macro')
        return -score

    def fit(self, oof_probs: np.ndarray, y_true: np.ndarray):
        n_classes = oof_probs.shape[1]
        init_weights = np.ones(n_classes)
        res = minimize(
            self._loss_func,
            init_weights,
            args=(oof_probs, y_true),
            method='Nelder-Mead',
            options={'maxiter': 500}
        )
        self.best_multipliers = res.x
        return self.best_multipliers

    def predict(self, probs: np.ndarray) -> np.ndarray:
        adjusted_probs = probs * self.best_multipliers
        return np.argmax(adjusted_probs, axis=1)
