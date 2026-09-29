"""Run the full pipeline end to end: data, tuning, intervals, final test, figures.

Usage: python -m src.pipeline
"""
from src import build_demo, download_data, plots, run_conformal, run_experiments, run_final


def main():
    download_data.main()
    run_experiments.main()
    run_conformal.main()
    run_final.main()
    plots.main()
    build_demo.main()


if __name__ == "__main__":
    main()
