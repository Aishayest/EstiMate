"""Trivial reference models that any text-based model must beat."""
import numpy as np


class MedianBaseline:
    """Predicts the median story point of the training set for every issue."""

    def fit(self, y_train):
        self.value_ = float(np.median(y_train))
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.value_)
