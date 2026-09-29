"""Fit and save the demo model: TF-IDF + Ridge with split conformal intervals.

Same protocol as the final evaluation (fit on train, calibrate on val), so the
demo behaves exactly like the model reported in results/final_test_*.csv.

Usage: python -m src.build_demo
"""
import json

import joblib
import numpy as np

from src import config
from src.conformal import conformal_quantile
from src.data import load_all
from src.models import make_ridge

MODELS_DIR = config.ROOT / "models"
DEMO_PATH = MODELS_DIR / "demo_ridge.joblib"
LEVELS = [0.5, 0.8, 0.9]


def main():
    best_params = json.loads((config.RESULTS_DIR / "best_params.json").read_text())
    df = load_all()
    bundle = {}
    for project, g in df.groupby("project", sort=False):
        train, cal = g[g.split == "train"], g[g.split == "val"]
        y_train, y_cal = train.storypoint.values, cal.storypoint.values
        model = make_ridge(**best_params[project]["tfidf_ridge"]).fit(train.text, y_train)
        lo, hi = float(y_train.min()), float(y_train.max())
        pred_cal = np.clip(model.predict(cal.text), lo, hi)
        residuals = np.abs(y_cal - pred_cal)
        bundle[project] = {
            "model": model,
            "clip": (lo, hi),
            "q": {level: conformal_quantile(residuals, 1 - level) for level in LEVELS},
            "scale": sorted(int(v) for v in np.unique(y_train)),
            "train_median": float(np.median(y_train)),
            "n_train": len(train),
            "n_cal": len(cal),
        }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, DEMO_PATH)
    print(f"Saved demo model for {len(bundle)} projects to {DEMO_PATH}")


if __name__ == "__main__":
    main()
