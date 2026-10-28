"""Overconfidence versus feature correlation — the mechanism behind the paradox.

As features become more correlated, the independence assumption is violated harder and
Naive Bayes double-counts evidence, so confidence climbs while accuracy barely moves.
Because the synthetic correlation is known, this is measured, not asserted.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.datasets import synthetic
from src.models import GaussianNB

from .probability import mean_confidence

Array = NDArray[np.float64]


def overconfidence_vs_correlation(
    correlations: list[float],
    n_samples: int = 2000,
    n_features: int = 20,
    class_sep: float = 1.5,
    seed: int = 42,
) -> tuple[list[float], list[float], list[float]]:
    """Return ``(correlations, mean_confidence, accuracy)`` across correlation strengths."""
    confidences: list[float] = []
    accuracies: list[float] = []
    for rho in correlations:
        X, y = synthetic.make_correlated_data(
            n_samples=n_samples,
            n_features=n_features,
            correlation=rho,
            class_sep=class_sep,
            seed=seed,
        )
        model = GaussianNB().fit(X, y)
        proba = model.predict_proba(X)
        confidences.append(mean_confidence(proba))
        accuracies.append(float((model.predict(X) == y).mean()))
    return list(correlations), confidences, accuracies
