"""Split conformal prediction and conformalized quantile regression (CQR).

All methods follow the same recipe: fit on a training set, compute
nonconformity scores on a separate, later calibration set, and take the
finite-sample-corrected (1 - alpha) quantile of those scores.
Coverage is guaranteed only if calibration and test issues are exchangeable;
with a chronological split this is an assumption we test, not a given.
"""
import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """ceil((n + 1)(1 - alpha)) / n empirical quantile of calibration scores."""
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return np.inf
    return float(np.quantile(scores, level, method="higher"))


def split_abs(pred_cal, y_cal, pred_test, alpha):
    """Constant-width interval: prediction +/- q of |y - y_hat|."""
    q = conformal_quantile(np.abs(y_cal - pred_cal), alpha)
    return pred_test - q, pred_test + q


def split_log(pred_cal, y_cal, pred_test, alpha):
    """Multiplicative interval: symmetric in log1p space, wider for larger estimates."""
    pred_cal, pred_test = np.maximum(pred_cal, 0), np.maximum(pred_test, 0)
    q = conformal_quantile(np.abs(np.log1p(y_cal) - np.log1p(pred_cal)), alpha)
    return np.expm1(np.log1p(pred_test) - q), np.expm1(np.log1p(pred_test) + q)


def cqr(lo_cal, hi_cal, y_cal, lo_test, hi_test, alpha):
    """Conformalized quantile regression (Romano et al., 2019)."""
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    q = conformal_quantile(scores, alpha)
    return lo_test - q, hi_test + q


def snap_to_integers(lo, hi, min_value: float = 1.0):
    """Round bounds inward to integers (story points are integer-valued).

    For integer labels y, lo <= y <= hi  <=>  ceil(lo) <= y <= floor(hi), so
    coverage is unchanged while the reported interval gets narrower. Empty
    intervals (no integer inside) collapse to the nearest integer of their
    midpoint, which can only increase coverage.
    """
    lo_i = np.maximum(np.ceil(lo), min_value)
    hi_i = np.floor(hi)
    empty = lo_i > hi_i
    mid = np.maximum(np.round((lo + hi) / 2), min_value)
    lo_i[empty], hi_i[empty] = mid[empty], mid[empty]
    return lo_i, hi_i


def coverage(y, lo, hi) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(hi - lo))
