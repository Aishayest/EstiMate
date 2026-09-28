"""Development-phase evaluation of prediction intervals (test split untouched).

Protocol per project, mirroring the final one (fit=train, cal=val, eval=test):
    fit  = oldest 75% of train
    cal  = newest 25% of train
    eval = val
Hyperparameters come from results/best_params.json (tuned on val), so point
accuracy on val is slightly optimistic; coverage depends on calibration only.

Usage: python -m src.run_conformal
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.conformal import coverage, cqr, mean_width, snap_to_integers, split_abs, split_log
from src.data import load_all
from src.embeddings import embed
from src.models import make_gbm, make_gbm_quantile, make_ridge

ALPHAS = [0.5, 0.2, 0.1]  # nominal coverage 50%, 80%, 90%
FIT_FRAC = 0.75


def inner_split(train: pd.DataFrame):
    train = train.sort_values("order")
    n_fit = int(len(train) * FIT_FRAC)
    return train.iloc[:n_fit], train.iloc[n_fit:]


def point_predictions(params, fit, cal, ev, E_fit, E_cal, E_ev):
    """Return {model_name: (pred_cal, pred_eval)} for every point model."""
    y_fit = fit.storypoint.values
    lo, hi = y_fit.min(), y_fit.max()
    clip = lambda p: np.clip(p, lo, hi)  # same post-processing as in tuning

    med = np.median(y_fit)
    ridge = make_ridge(**params["tfidf_ridge"]).fit(fit.text, y_fit)
    gbm = make_gbm(**params["minilm_hgb"]).fit(E_fit, y_fit)
    return {
        "median_train": (np.full(len(cal), med), np.full(len(ev), med)),
        "tfidf_ridge": (clip(ridge.predict(cal.text)), clip(ridge.predict(ev.text))),
        "minilm_hgb": (clip(gbm.predict(E_cal)), clip(gbm.predict(E_ev))),
    }


def main():
    best_params = json.loads((config.RESULTS_DIR / "best_params.json").read_text())
    df = load_all()
    rows = []

    for project, g in df.groupby("project", sort=False):
        fit, cal = inner_split(g[g.split == "train"])
        ev = g[g.split == "val"]
        y_fit, y_cal, y_ev = fit.storypoint.values, cal.storypoint.values, ev.storypoint.values

        # Cached embeddings of the full train split, sliced into fit/cal.
        train_sorted = g[g.split == "train"].sort_values("order")
        E_train = embed(list(train_sorted.text))
        E_fit, E_cal = E_train[: len(fit)], E_train[len(fit):]
        E_ev = embed(list(ev.text))

        params = best_params[project]
        preds = point_predictions(params, fit, cal, ev, E_fit, E_cal, E_ev)

        for alpha in ALPHAS:
            split_methods = {
                "median+split": ("median_train", split_abs),
                "ridge+split": ("tfidf_ridge", split_abs),
                "ridge+split_log": ("tfidf_ridge", split_log),
                "minilm_hgb+split": ("minilm_hgb", split_abs),
            }
            intervals = {}
            for method, (model, fn) in split_methods.items():
                pred_cal, pred_ev = preds[model]
                intervals[method] = fn(pred_cal, y_cal, pred_ev, alpha)
            q_lo = make_gbm_quantile(alpha / 2, **params["minilm_hgb"]).fit(E_fit, y_fit)
            q_hi = make_gbm_quantile(1 - alpha / 2, **params["minilm_hgb"]).fit(E_fit, y_fit)
            intervals["minilm_hgb+cqr"] = cqr(
                q_lo.predict(E_cal), q_hi.predict(E_cal), y_cal,
                q_lo.predict(E_ev), q_hi.predict(E_ev), alpha,
            )

            for method, (lo, hi) in intervals.items():
                lo_i, hi_i = snap_to_integers(lo, hi)
                rows.append({
                    "project": project,
                    "method": method,
                    "nominal": 1 - alpha,
                    "coverage": round(coverage(y_ev, lo_i, hi_i), 3),
                    "mean_width": round(mean_width(lo_i, hi_i), 2),
                    "median_width": float(np.median(hi_i - lo_i)),
                    "n_cal": len(cal),
                    "n_eval": len(ev),
                })

    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "conformal_dev.csv", index=False)
    for nominal, r in res.groupby("nominal"):
        print(f"\n=== nominal coverage {nominal:.0%} ===")
        print(r.pivot(index="method", columns="project", values=["coverage", "mean_width"]).round(2))


if __name__ == "__main__":
    main()
