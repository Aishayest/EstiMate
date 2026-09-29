"""Sentence-transformer embeddings with an on-disk cache.

The encoder is pretrained and frozen (no fine-tuning), so embedding every
split with it does not leak information from val/test into training.
"""
import hashlib

import numpy as np

from src import config

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned Hugging Face commit, so the same weights are used on every run.
EMBED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CACHE_DIR = config.ROOT / "data" / "processed" / "embeddings"


def embed(texts: list[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Return L2-normalised embeddings, reusing a cache keyed by model and texts."""
    key = hashlib.sha256(
        "\x00".join([model_name, EMBED_REVISION, *texts]).encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / f"{model_name.split('/')[-1]}_{key}.npy"
    if path.exists():
        return np.load(path)

    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    model = SentenceTransformer(model_name, revision=EMBED_REVISION, device="cpu")
    X = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    return X
