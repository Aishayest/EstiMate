"""Final, one-shot evaluation on the held-out test split.

Protocol per project: fit on train, calibrate intervals on val, evaluate on test.
Models, interval methods and coverage levels were fixed before test was opened:
    point:     median_train, tfidf_ridge, minilm_hgb
    intervals: ridge+split (primary), median+split (reference), minilm_hgb+cqr
    levels:    50%, 80%, 90%
Hyperparameters come from results/best_params.json (tuned on val).

Usage: python -m src.run_final
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.conformal import coverage, cqr, mean_width, snap_to_integers, split_abs
from src.data import load_all
from src.embeddings import embed
from src.metrics import mae, mdae
from src.models import make_gbm, make_gbm_quantile, make_ridge

ALPHAS = [0.5, 0.2, 0.1]
N_BOOT = 2000
N_TIME_BINS = 4


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean of per-issue values."""
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rng = np.random.default_rng(config.SEED)
    best_params = json.loads((config.RESULTS_DIR / "best_params.json").read_text())
    df = load_all()
    point_rows, interval_rows, time_rows, pred_frames = [], [], [], []

    for project, g in df.groupby("project", sort=False):
        train, cal, test = (g[g.split == s].sort_values("order") for s in ["train", "val", "test"])
        y_train, y_cal, y_test = (d.storypoint.values for d in (train, cal, test))
        E_train, E_cal, E_test = (embed(list(d.text)) for d in (train, cal, test))
        params = best_params[project]
        clip = lambda p: np.clip(p, y_train.min(), y_train.max())

        # Point models, fit on train only (the same fits are used for intervals).
        ridge = make_ridge(**params["tfidf_ridge"]).fit(train.text, y_train)
        gbm = make_gbm(**params["minilm_hgb"]).fit(E_train, y_train)
        med = np.median(y_train)
        preds = {
            "median_train": (np.full(len(cal), med), np.full(len(test), med)),
            "tfidf_ridge": (clip(ridge.predict(cal.text)), clip(ridge.predict(test.text))),
            "minilm_hgb": (clip(gbm.predict(E_cal)), clip(gbm.predict(E_test))),
        }
        for model, (_, pred) in preds.items():
            abs_err = np.abs(y_test - pred)
            lo, hi = bootstrap_ci(abs_err, rng)
            point_rows.append({
                "project": project, "model": model,
                "test_mae": round(mae(y_test, pred), 3),
                "mae_ci_low": round(lo, 3), "mae_ci_high": round(hi, 3),
                "test_mdae": round(mdae(y_test, pred), 3),
                "n_test": len(test),
            })

        pred_frame = test[["project", "issuekey", "order", "storypoint"]].copy()
        pred_frame["pred_ridge"] = preds["tfidf_ridge"][1]
        time_bin = np.arange(len(test)) * N_TIME_BINS // len(test)

        for alpha in ALPHAS:
            q_lo = make_gbm_quantile(alpha / 2, **params["minilm_hgb"]).fit(E_train, y_train)
            q_hi = make_gbm_quantile(1 - alpha / 2, **params["minilm_hgb"]).fit(E_train, y_train)
            intervals = {
                "ridge+split": split_abs(preds["tfidf_ridge"][0], y_cal, preds["tfidf_ridge"][1], alpha),
                "median+split": split_abs(preds["median_train"][0], y_cal, preds["median_train"][1], alpha),
                "minilm_hgb+cqr": cqr(q_lo.predict(E_cal), q_hi.predict(E_cal), y_cal,
                                      q_lo.predict(E_test), q_hi.predict(E_test), alpha),
            }
            for method, (lo, hi) in intervals.items():
                lo_i, hi_i = snap_to_integers(lo, hi)
                covered = ((y_test >= lo_i) & (y_test <= hi_i)).astype(float)
                c_lo, c_hi = bootstrap_ci(covered, rng)
                interval_rows.append({
                    "project": project, "method": method, "nominal": 1 - alpha,
                    "coverage": round(coverage(y_test, lo_i, hi_i), 3),
                    "coverage_ci_low": round(c_lo, 3), "coverage_ci_high": round(c_hi, 3),
                    "mean_width": round(mean_width(lo_i, hi_i), 2),
                    "median_width": float(np.median(hi_i - lo_i)),
                    "n_cal": len(cal), "n_test": len(test),
                })
                for b in range(N_TIME_BINS):
                    m = time_bin == b
                    time_rows.append({
                        "project": project, "method": method, "nominal": 1 - alpha,
                        "time_bin": b + 1, "coverage": round(covered[m].mean(), 3), "n": int(m.sum()),
                    })
                if method == "ridge+split":
                    pred_frame[f"lo_{1 - alpha:.0%}"] = lo_i
                    pred_frame[f"hi_{1 - alpha:.0%}"] = hi_i
        pred_frames.append(pred_frame)

    out = config.RESULTS_DIR
    points, intervals = pd.DataFrame(point_rows), pd.DataFrame(interval_rows)
    points.to_csv(out / "final_test_point.csv", index=False)
    intervals.to_csv(out / "final_test_intervals.csv", index=False)
    pd.DataFrame(time_rows).to_csv(out / "final_test_coverage_over_time.csv", index=False)
    pd.concat(pred_frames).to_csv(out / "final_test_predictions.csv", index=False)

    print(points.to_string(index=False))
    print(intervals[intervals.nominal == 0.9].to_string(index=False))


if __name__ == "__main__":
    main()
