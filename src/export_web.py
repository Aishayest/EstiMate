"""Export the demo model and test results as static JSON for the web frontend.

The browser re-implements TF-IDF + Ridge inference, so each project file holds
the fitted vocabulary, IDF weights, Ridge coefficients, conformal margins q and
the train issues (with their TF-IDF vectors) used for "nearest past issues".

Usage: python -m src.export_web   (after src.build_demo and src.run_final)
"""
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

from src import config
from src.build_demo import DEMO_PATH
from src.data import load_all

WEB_DIR = config.ROOT / "web"
ARTIFACTS_DIR = WEB_DIR / "artifacts"
N_EXAMPLES = 30
DRIFT_GAP = 0.05  # coverage this far below target at any level marks drift


def unpack(model):
    """Return (vectorizer, ridge, log_target) from a fitted demo model."""
    log_target = isinstance(model, TransformedTargetRegressor)
    pipe = model.regressor_ if log_target else model
    return pipe.named_steps["tfidfvectorizer"], pipe.named_steps["ridge"], log_target


def sparse_rows(X, decimals: int = 4) -> list[list]:
    """CSR matrix -> per-row [[indices], [values]] with rounded values."""
    rows = []
    for i in range(X.shape[0]):
        s, e = X.indptr[i], X.indptr[i + 1]
        rows.append([X.indices[s:e].tolist(), np.round(X.data[s:e], decimals).tolist()])
    return rows


def project_artifact(project, entry, train, test, rng):
    vec, ridge, log_target = unpack(entry["model"])
    terms = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    examples = test.sample(min(N_EXAMPLES, len(test)), random_state=rng)
    return {
        "project": project,
        "tfidf": {
            "ngram_range": list(vec.ngram_range),
            "terms": terms,
            "idf": np.round(vec.idf_, 6).tolist(),
        },
        "ridge": {
            "coef": np.round(ridge.coef_, 8).tolist(),
            "intercept": float(ridge.intercept_),
            "log_target": log_target,
        },
        "clip": list(entry["clip"]),
        "q": {f"{int(level * 100)}": float(q) for level, q in entry["q"].items()},
        "scale": entry["scale"],
        "train": {
            "keys": train.issuekey.tolist(),
            "titles": train.title.tolist(),
            "sp": train.storypoint.astype(int).tolist(),
            "vectors": sparse_rows(vec.transform(train.text)),
        },
        "examples": [
            {"key": r.issuekey, "title": r.title, "description": r.description,
             "sp": int(r.storypoint)}
            for r in examples.itertuples()
        ],
    }


def summary(df, bundle):
    points = pd.read_csv(config.RESULTS_DIR / "final_test_point.csv")
    intervals = pd.read_csv(config.RESULTS_DIR / "final_test_intervals.csv")
    intervals = intervals[intervals.method == "ridge+split"]
    projects = {}
    for project in bundle:
        pt = points[points.project == project].set_index("model")
        iv = intervals[intervals.project == project].set_index("nominal")
        g = df[df.project == project]
        coverage = {
            f"{int(n * 100)}": {
                "coverage": float(r.coverage), "ci": [float(r.coverage_ci_low), float(r.coverage_ci_high)],
                "mean_width": float(r.mean_width),
            }
            for n, r in iv.iterrows()
        }
        mae, base = float(pt.loc["tfidf_ridge", "test_mae"]), float(pt.loc["median_train", "test_mae"])
        projects[project] = {
            "n_train": int((g.split == "train").sum()),
            "n_cal": int((g.split == "val").sum()),
            "n_test": int((g.split == "test").sum()),
            "mae": mae,
            "mae_ci": [float(pt.loc["tfidf_ridge", "mae_ci_low"]), float(pt.loc["tfidf_ridge", "mae_ci_high"])],
            "mae_median_baseline": base,
            "gain_vs_median": round(1 - mae / base, 4),
            "median_sp_train": float(g[g.split == "train"].storypoint.median()),
            "median_sp_test": float(g[g.split == "test"].storypoint.median()),
            "coverage": coverage,
            "drift": bool((iv.coverage < iv.index - DRIFT_GAP).any()),
        }
    # Overall numbers from per-issue predictions, so they are not means of rounded values.
    preds = pd.read_csv(config.RESULTS_DIR / "final_test_predictions.csv")
    covered = (preds.storypoint >= preds["lo_90%"]) & (preds.storypoint <= preds["hi_90%"])
    per_project = preds.assign(covered=covered, width=preds["hi_90%"] - preds["lo_90%"]).groupby("project")
    return {
        "method": "TF-IDF + Ridge, split conformal",
        "n_issues": int(len(df)),
        "projects": projects,
        "overall": {
            "coverage_90": round(float(per_project.covered.mean().mean()), 4),
            "mean_width_90": round(float(per_project.width.mean().mean()), 2),
        },
    }


def main():
    bundle = joblib.load(DEMO_PATH)
    df = load_all()
    rng = np.random.RandomState(config.SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for project, entry in bundle.items():
        g = df[df.project == project]
        art = project_artifact(project, entry, g[g.split == "train"], g[g.split == "test"], rng)
        path = ARTIFACTS_DIR / f"{project}.json"
        path.write_text(json.dumps(art, separators=(",", ":"), ensure_ascii=False))
        print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB")
    (ARTIFACTS_DIR / "summary.json").write_text(json.dumps(summary(df, bundle), indent=2))

    fig_dir = WEB_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    for png in (config.RESULTS_DIR / "figures").glob("*.png"):
        shutil.copy(png, fig_dir / png.name)


if __name__ == "__main__":
    main()
