"""Check that the browser model (web/assets/estimator.js) matches Python on all test issues.

Usage: python tests/web_parity.py   (needs node; run after src.export_web)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.build_demo import DEMO_PATH  # noqa: E402
from src.conformal import snap_to_integers  # noqa: E402
from src.data import load_all  # noqa: E402


def main():
    bundle = joblib.load(DEMO_PATH)
    df = load_all()
    ref = {}
    for project, entry in bundle.items():
        test = df[(df.project == project) & (df.split == "test")]
        point = np.clip(entry["model"].predict(test.text), *entry["clip"])
        intervals = {}
        for level, q in entry["q"].items():
            lo, hi = snap_to_integers(point - q, point + q)
            intervals[int(level * 100)] = np.stack([lo, hi], axis=1).astype(int).tolist()
        ref[project] = [
            {"text": t, "point": float(p), "intervals": {lvl: iv[i] for lvl, iv in intervals.items()}}
            for i, (t, p) in enumerate(zip(test.text, point))
        ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(ref, f)
    script = Path(__file__).with_name("parity.cjs")
    sys.exit(subprocess.run(["node", str(script), f.name]).returncode)


if __name__ == "__main__":
    main()
