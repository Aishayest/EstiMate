"""Project-wide configuration: paths, selected projects, split ratios, seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

# Four projects from four different organisations, each with >1300 issues and a
# standard (Fibonacci-like) story point scale. datamanagement and moodle are
# excluded because of heavy-tailed, non-standard scales (values up to 100).
PROJECTS = ["mesos", "springxd", "appceleratorstudio", "talenddataquality"]

# Chronological split within each project (rows are already sorted by creation time).
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2  # test gets the remaining 0.2

SEED = 42
