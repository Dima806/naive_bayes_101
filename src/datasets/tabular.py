"""Tabular datasets for Gaussian Naive Bayes and the general comparison.

Uses scikit-learn's bundled datasets (no network), plus a categorical variant obtained
by binning the continuous features — so Bernoulli/Multinomial variants have something
to chew on too.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer

Array = NDArray[np.float64]
Labels = NDArray[np.int64]


def load_continuous() -> tuple[Array, Labels, list[str]]:
    """Continuous features (breast cancer): the natural home of Gaussian Naive Bayes."""
    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)
    return X, y, list(data.feature_names)


def load_categorical(n_bins: int = 4) -> tuple[Array, Labels, list[str]]:
    """The same data binned into ordinal categories, for the count/binary variants."""
    X, y, names = load_continuous()
    binned = np.zeros_like(X)
    for j in range(X.shape[1]):
        edges = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        binned[:, j] = np.digitize(X[:, j], edges)
    return binned, y, names
