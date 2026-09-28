"""Loading, cleaning and chronological splitting of the Deep-SE story point data."""
import pandas as pd

from src import config


def load_project(name: str) -> pd.DataFrame:
    """Read one project's CSV, keeping the original row order (= creation time order)."""
    df = pd.read_csv(config.RAW_DIR / f"{name}.csv")
    df["project"] = name
    df["order"] = range(len(df))  # explicit time index; never sort by issue key
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise text, drop empty issues and within-project duplicate texts.

    Duplicates are dropped keeping the earliest occurrence, so the same issue
    text can never appear in both train and a later split.
    """
    df = df.copy()
    for col in ["title", "description"]:
        df[col] = df[col].fillna("").astype(str).str.split().str.join(" ")
    df["text"] = (df["title"] + " " + df["description"]).str.strip()

    n_before = len(df)
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["project", "text"], keep="first")
    df.attrs["n_dropped"] = n_before - len(df)
    return df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test by position in time within a single project."""
    df = df.sort_values("order").reset_index(drop=True)
    n = len(df)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    split = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    return df.assign(split=split)


def load_all(projects: list[str] = config.PROJECTS) -> pd.DataFrame:
    """Load, clean and split all selected projects into one long DataFrame."""
    parts = []
    for name in projects:
        df = clean(load_project(name))
        parts.append(temporal_split(df))
    return pd.concat(parts, ignore_index=True)
