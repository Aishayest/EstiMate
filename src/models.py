"""Text regression models and validation-set hyperparameter search.

Every model is fit on the train split only; hyperparameters are chosen by MAE
on the val split. The test split is never seen here.
"""
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from src import config
from src.metrics import mae

RIDGE_GRID = {
    "ngram_max": [1, 2],
    "alpha": [0.3, 1.0, 3.0, 10.0, 30.0],
    "log_target": [False, True],
}


def make_ridge(ngram_max: int, alpha: float, log_target: bool):
    # The vectorizer lives inside the pipeline, so its vocabulary and IDF
    # weights are learned from the training texts only.
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, ngram_max), min_df=2, sublinear_tf=True),
        Ridge(alpha=alpha, random_state=config.SEED),
    )
    if log_target:
        return TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


GBM_GRID = {
    "loss": ["squared_error", "absolute_error"],
    "learning_rate": [0.03, 0.1],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],
}


def make_gbm(loss: str, learning_rate: float, max_leaf_nodes: int, min_samples_leaf: int):
    # Histogram gradient boosting (LightGBM-style). Built-in early stopping is
    # disabled because it would carve a *random* validation subset out of train.
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        max_iter=300,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=config.SEED,
    )


def make_gbm_quantile(quantile: float, learning_rate: float, max_leaf_nodes: int,
                      min_samples_leaf: int, **_):
    """Quantile-loss variant of make_gbm, reusing the tuned tree hyperparameters."""
    model = make_gbm("absolute_error", learning_rate, max_leaf_nodes, min_samples_leaf)
    return model.set_params(loss="quantile", quantile=quantile)


def grid_search(make_model, grid: dict, X_train, y_train, X_val, y_val):
    """Fit one model per grid point on train, score on val, return the best.

    Predictions are clipped to the range of training labels, a harmless
    post-processing step that uses train information only.
    """
    lo, hi = float(np.min(y_train)), float(np.max(y_train))
    rows, best = [], None
    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        model = make_model(**params).fit(X_train, y_train)
        pred = np.clip(model.predict(X_val), lo, hi)
        score = mae(y_val, pred)
        rows.append({**params, "val_mae": score})
        if best is None or score < best[0]:
            best = (score, params, pred)
    return best[1], best[2], pd.DataFrame(rows)
