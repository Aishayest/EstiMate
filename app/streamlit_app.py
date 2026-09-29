"""EstiMate demo: story point estimate with a calibrated prediction interval.

Run:
    python -m src.build_demo            # once, fits and saves the model
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.build_demo import DEMO_PATH  # noqa: E402
from src.conformal import snap_to_integers  # noqa: E402
from src.data import load_all  # noqa: E402

st.set_page_config(page_title="EstiMate", page_icon="🎯", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load(DEMO_PATH)


@st.cache_data
def load_test_report() -> pd.DataFrame:
    r = pd.read_csv(config.RESULTS_DIR / "final_test_intervals.csv")
    return r[r.method == "ridge+split"]


@st.cache_data
def load_test_issues() -> pd.DataFrame:
    df = load_all()
    return df[df.split == "test"][["project", "issuekey", "title", "description", "storypoint"]]


def predict(entry: dict, text: str, level: float):
    point = float(np.clip(entry["model"].predict([text])[0], *entry["clip"]))
    q = entry["q"][level]
    lo, hi = snap_to_integers(np.array([point - q]), np.array([point + q]))
    return point, int(lo[0]), int(hi[0])


def scale_strip(scale: list[int], lo: int, hi: int, truth: int | None) -> str:
    """Project's story point scale with values inside the interval highlighted."""
    chips = []
    for v in scale:
        inside = lo <= v <= hi
        style = (
            "background:#2a78d6;color:#fff;border:1px solid #2a78d6;"
            if inside else "background:transparent;color:#52514e;border:1px solid #d4d3cf;"
        )
        mark = " ✓" if truth is not None and v == truth else ""
        chips.append(
            f'<span style="{style}padding:4px 10px;border-radius:12px;margin:2px;'
            f'display:inline-block;font-size:0.9rem;">{v}{mark}</span>'
        )
    return "<div>" + "".join(chips) + "</div>"


bundle = load_bundle()
report = load_test_report()

st.title("EstiMate")
st.caption(
    "Story point estimate from issue text, with a conformal prediction interval. "
    "Model: TF-IDF + Ridge, calibrated on the validation period of each project."
)

with st.sidebar:
    project = st.selectbox("Project", list(bundle))
    level = st.radio("Interval coverage", [0.9, 0.8, 0.5], format_func=lambda x: f"{x:.0%}")
    if st.button("Load a random test issue"):
        issues = load_test_issues()
        row = issues[issues.project == project].sample(1).iloc[0]
        st.session_state.update(
            title=row.title, description=row.description,
            truth=int(row.storypoint), issuekey=row.issuekey,
            example_text=f"{row.title} {row.description}".strip(),
        )

title = st.text_input("Title", key="title")
description = st.text_area("Description", key="description", height=180)
text = f"{title} {description}".strip()

# Forget the true label once the user edits the loaded example.
if st.session_state.get("example_text") not in (None, text):
    for key in ("truth", "issuekey", "example_text"):
        st.session_state.pop(key, None)

if not text:
    st.info("Enter an issue title and/or description, or load a test issue from the sidebar.")
    st.stop()

entry = bundle[project]
point, lo, hi = predict(entry, text, level)
truth = st.session_state.get("truth")

c1, c2 = st.columns(2)
c1.metric("Point estimate", f"{point:.1f} SP")
c2.metric(f"{level:.0%} interval", f"{lo}–{hi} SP" if lo != hi else f"{lo} SP")
st.markdown(scale_strip(entry["scale"], lo, hi, truth), unsafe_allow_html=True)

if truth is not None:
    verdict = "inside" if lo <= truth <= hi else "outside"
    st.write(f"True estimate for **{st.session_state['issuekey']}**: **{truth} SP**, {verdict} the interval.")

r = report[(report.project == project) & (report.nominal == level)].iloc[0]
st.divider()
st.markdown(
    f"**How reliable is this on {project}?** On {int(r.n_test)} held-out test issues, "
    f"{level:.0%} intervals contained the true value **{r.coverage:.0%}** of the time "
    f"(95% CI {r.coverage_ci_low:.0%}–{r.coverage_ci_high:.0%}), mean width {r.mean_width:.1f} SP."
)
if r.coverage < level - 0.05:
    st.warning(
        "Coverage on test fell clearly below the target for this project and level. "
        "Estimates in this project drifted over time, so the calibration period no "
        "longer represents new issues well."
    )
st.caption(
    f"Trained on {entry['n_train']} issues, calibrated on {entry['n_cal']}. "
    "Point accuracy from text alone is modest (about 5–8% better MAE than always "
    "predicting the training median), so treat the interval as the main output."
)
