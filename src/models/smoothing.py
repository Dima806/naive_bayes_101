"""Laplace / additive smoothing — the fix that makes Naive Bayes usable.

Without smoothing, a feature value that never appeared in a class during training
gets probability 0. Because Naive Bayes multiplies per-feature probabilities (adds
log-probabilities), a single 0 annihilates the whole prediction: ``log(0) = -inf``.

Additive smoothing adds a pseudo-count ``alpha`` to every count so no probability is
ever exactly 0. ``alpha = 1`` is classic Laplace smoothing. ``alpha = 0`` recovers the
unsmoothed (breakable) model and is kept only to *demonstrate* the failure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def multinomial_feature_log_prob(feature_count: Array, alpha: float) -> Array:
    """Smoothed ``log P(feature | class)`` for Multinomial Naive Bayes.

    ``feature_count`` has shape ``(n_classes, n_features)``. Each class row is
    normalised over features, so ``exp(result).sum(axis=1) == 1`` per class.
    """
    smoothed_fc = feature_count + alpha
    smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(smoothed_fc) - np.log(smoothed_cc)


def bernoulli_feature_log_prob(feature_count: Array, class_count: Array, alpha: float) -> Array:
    """Smoothed ``log P(feature present | class)`` for Bernoulli Naive Bayes.

    Each feature is an independent present/absent Bernoulli, so the denominator is
    the class size plus ``2 * alpha`` (one pseudo-count each for present and absent).
    """
    smoothed_fc = feature_count + alpha
    smoothed_cc = class_count + alpha * 2
    with np.errstate(divide="ignore"):
        return np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))
