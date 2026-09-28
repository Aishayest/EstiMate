"""Train all models per project, tune on val, and write a summary table.

Usage: python -m src.run_experiments
"""
import pandas as pd

from src import config
from src.baselines import MedianBaseline
from src.data import load_all
from src.embeddings import embed
from src.metrics import mae, mdae
from src.models import GBM_GRID, RIDGE_GRID, grid_search, make_gbm, make_ridge


def evaluate(project, model_name, y_val, pred, params=None):
    return {
        "project": project,
        "model": model_name,
        "val_mae": round(mae(y_val, pred), 3),
        "val_mdae": round(mdae(y_val, pred), 3),
        "best_params": params or {},
    }


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    df = load_all()
    df.groupby(["project", "split"]).size().unstack()[["train", "val", "test"]].to_csv(
        config.RESULTS_DIR / "split_sizes.csv"
    )

    rows, tuning_logs = [], []
    for project, g in df.groupby("project", sort=False):
        train, val = g[g.split == "train"], g[g.split == "val"]
        y_train, y_val = train.storypoint.values, val.storypoint.values

        base = MedianBaseline().fit(y_train)
        rows.append(evaluate(project, "median_train", y_val, base.predict(len(val))))

        params, pred, log = grid_search(
            make_ridge, RIDGE_GRID, train.text, y_train, val.text, y_val
        )
        rows.append(evaluate(project, "tfidf_ridge", y_val, pred, params))
        tuning_logs.append(log.assign(project=project, model="tfidf_ridge"))

        # Only train and val texts are embedded; test stays untouched until the end.
        E_train, E_val = embed(list(train.text)), embed(list(val.text))
        params, pred, log = grid_search(make_gbm, GBM_GRID, E_train, y_train, E_val, y_val)
        rows.append(evaluate(project, "minilm_hgb", y_val, pred, params))
        tuning_logs.append(log.assign(project=project, model="minilm_hgb"))

    summary = pd.DataFrame(rows)
    summary.to_csv(config.RESULTS_DIR / "summary_val.csv", index=False)
    pd.concat(tuning_logs).to_csv(config.RESULTS_DIR / "tuning_log.csv", index=False)

    pivot = summary.pivot(index="project", columns="model", values="val_mae")
    print(summary.to_string(index=False), "\n\nVal MAE:\n", pivot)


if __name__ == "__main__":
    main()
