"""Run the full development pipeline end to end.

Usage: python -m src.pipeline
"""
from src import download_data, run_conformal, run_experiments


def main():
    download_data.main()
    run_experiments.main()
    run_conformal.main()


if __name__ == "__main__":
    main()
