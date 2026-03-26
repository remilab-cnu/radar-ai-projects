"""Evaluation metrics shared across projects."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error,
)


# -- Classification metrics --------------------------------------------------

def classification_report(y_true, y_pred, y_prob=None, labels=None) -> dict:
    """Compute standard classification metrics."""
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        m["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return m


def pd_at_pfa(y_true, y_score, target_pfa: float) -> float:
    """Compute Pd at a given Pfa from scores."""
    neg_scores = y_score[y_true == 0]
    pos_scores = y_score[y_true == 1]
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return 0.0
    threshold = np.percentile(neg_scores, 100 * (1 - target_pfa))
    return float(np.mean(pos_scores >= threshold))


# -- Regression metrics -------------------------------------------------------

def regression_report(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# -- Signal quality metrics ---------------------------------------------------

def psnr(target: np.ndarray, prediction: np.ndarray, max_val: float | None = None) -> float:
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        return float("inf")
    if max_val is None:
        max_val = target.max() - target.min()
    return float(10 * np.log10(max_val ** 2 / mse))


def nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target - prediction) ** 2) / np.mean(target ** 2))
