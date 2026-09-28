"""Evaluation metrics for point estimates (and, later, prediction intervals)."""
import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mdae(y_true, y_pred) -> float:
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
