"""Figures for the final test evaluation, saved as PNG into results/figures/.

Usage: python -m src.plots
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

FIG_DIR = config.RESULTS_DIR / "figures"

# Validated categorical palette (fixed order); markers give a second,
# colour-independent encoding for each method.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHODS = {
    "ridge+split": ("TF-IDF + Ridge, split conformal", "#2a78d6", "o"),
    "minilm_hgb+cqr": ("MiniLM + boosting, CQR", "#eb6834", "s"),
    "median+split": ("Median baseline, split conformal", "#1baf7a", "^"),
}
PROJECTS = ["mesos", "springxd", "appceleratorstudio", "talenddataquality"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def calibration_plot(intervals: pd.DataFrame):
    """Empirical vs nominal coverage per project; the diagonal is perfect calibration."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, project in zip(axes, PROJECTS):
        ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--", zorder=1)
        for method, (label, color, marker) in METHODS.items():
            r = intervals[(intervals.project == project) & (intervals.method == method)]
            r = r.sort_values("nominal")
            ax.plot(r.nominal, r.coverage, color=color, lw=2, marker=marker, ms=7,
                    mec=SURFACE, mew=1.5, label=label, zorder=2)
        ax.set_title(project, loc="left")
        ax.set_xticks([0.5, 0.8, 0.9])
        ax.set_xticklabels(["50%", "80%", "90%"])
        ax.set_xlim(0.45, 0.95)
        ax.set_ylim(0.2, 1.02)
        ax.set_xlabel("Target coverage")
    axes[0].set_ylabel("Empirical coverage on test")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def width_plot(intervals: pd.DataFrame):
    """Mean width of 90% intervals: narrower is better at equal coverage."""
    r = intervals[intervals.nominal == 0.9]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    n_methods, bar_h = len(METHODS), 0.26
    for i, (method, (label, color, _)) in enumerate(METHODS.items()):
        m = r[r.method == method].set_index("project").loc[PROJECTS]
        ys = [p - (i - 1) * bar_h for p in range(len(PROJECTS))]
        ax.barh(ys, m.mean_width, height=bar_h - 0.04, color=color, label=label)
        for y, w, c in zip(ys, m.mean_width, m.coverage):
            ax.text(w + 0.15, y, f"{w:.1f} SP  ({c:.0%})", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(PROJECTS)))
    ax.set_yticklabels(PROJECTS)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean interval width, story points (coverage in brackets)")
    ax.set_title("90% prediction intervals on test", loc="left")
    ax.set_xlim(0, r.mean_width.max() * 1.35)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "interval_width.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def coverage_over_time_plot(over_time: pd.DataFrame):
    """Coverage of 90% ridge intervals across chronological quarters of test."""
    r = over_time[(over_time.method == "ridge+split") & (over_time.nominal == 0.9)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    color = METHODS["ridge+split"][1]
    for ax, project in zip(axes, PROJECTS):
        p = r[r.project == project].sort_values("time_bin")
        ax.axhline(0.9, color=INK_2, lw=1, ls="--")
        ax.plot(p.time_bin, p.coverage, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
        ax.set_title(project, loc="left")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("Test quarter (older → newer)")
        ax.set_ylim(0.75, 1.02)
    axes[0].set_ylabel("Coverage")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].text(1, 0.905, "target 90%", fontsize=8, color=INK_2, va="bottom")
    fig.suptitle("TF-IDF + Ridge, 90% split conformal: coverage over time", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_over_time.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def example_intervals_plot(preds: pd.DataFrame, project: str = "mesos", n: int = 40):
    """Intervals for the n most recent test issues of one project."""
    p = preds[preds.project == project].sort_values("order").tail(n).reset_index(drop=True)
    color = METHODS["ridge+split"][1]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.vlines(p.index, p["lo_90%"], p["hi_90%"], color=color, lw=6, alpha=0.3,
              label="90% interval")
    ax.vlines(p.index, p["lo_50%"], p["hi_50%"], color=color, lw=6, label="50% interval")
    inside = (p.storypoint >= p["lo_90%"]) & (p.storypoint <= p["hi_90%"])
    ax.scatter(p.index[inside], p.storypoint[inside], color=INK, s=28, zorder=3,
               label="True story points")
    ax.scatter(p.index[~inside], p.storypoint[~inside], color=INK, marker="x", s=40,
               zorder=3, label="True, outside 90% interval")
    ax.set_xticks(p.index)
    ax.set_xticklabels(p.issuekey, rotation=90, fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Story points")
    ax.set_title(f"{project}: {n} most recent test issues", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    ax.set_ylim(0, max(p["hi_90%"].max(), p.storypoint.max()) * 1.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_intervals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(config.RESULTS_DIR / "final_test_intervals.csv")
    calibration_plot(intervals)
    width_plot(intervals)
    coverage_over_time_plot(pd.read_csv(config.RESULTS_DIR / "final_test_coverage_over_time.csv"))
    example_intervals_plot(pd.read_csv(config.RESULTS_DIR / "final_test_predictions.csv"))
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
