"""Synthetic data with two knobs that drive the project's arguments.

Because the structure is *known*, the claims are measured, not asserted:

* **correlation** — how strongly features share a common component. At ``0.0`` the
  naive independence assumption holds; as it rises toward ``1.0`` the assumption is
  violated and Naive Bayes grows overconfident. ``duplicate_feature`` gives the
  worst case: a perfectly correlated copy.
* **sample size** — small training sets are where the simple model beats the flexible
  one (logistic regression overfits, Naive Bayes does not).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Labels = NDArray[np.int64]


def make_correlated_data(
    n_samples: int = 2000,
    n_features: int = 20,
    correlation: float = 0.0,
    class_sep: float = 1.5,
    seed: int = 42,
) -> tuple[Array, Labels]:
    """Two-class Gaussian data with tunable per-class feature correlation.

    Each feature is ``class_shift + sqrt(rho) * shared + sqrt(1 - rho) * independent``.
    ``rho`` (``correlation``) controls how much a single shared component is baked into
    every feature: at ``rho = 0`` features are conditionally independent; at ``rho = 1``
    they collapse onto one component (maximally redundant).
    """
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [0, 1]")
    rng = np.random.default_rng(seed)

    y = rng.integers(0, 2, size=n_samples)
    shift = np.where(y == 1, class_sep / 2.0, -class_sep / 2.0).astype(np.float64)

    shared = rng.standard_normal((n_samples, 1))
    independent = rng.standard_normal((n_samples, n_features))
    X = shift[:, None] + np.sqrt(correlation) * shared + np.sqrt(1.0 - correlation) * independent
    return X, y.astype(np.int64)


def duplicate_feature(X: Array, feature_index: int = 0, n_copies: int = 1) -> Array:
    """Append ``n_copies`` perfect duplicates of one feature — the worst case for independence.

    Naive Bayes treats each copy as fresh evidence and double-counts it, so confidence
    climbs while the real information (and thus accuracy) is unchanged.
    """
    column = X[:, feature_index : feature_index + 1]
    extra = np.repeat(column, n_copies, axis=1)
    return np.hstack([X, extra])


def make_informative_plus_noise(
    n_samples: int = 800,
    n_noise: int = 3,
    signal: float = 0.8,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> tuple[Array, Labels]:
    """One genuinely informative feature (column 0) plus pure-noise features.

    The informative feature alone decides the class, so *duplicating* it (via
    :func:`duplicate_feature`) sharpens the posterior toward 0/1 without changing which
    class wins — the cleanest demonstration of overconfidence rising while accuracy holds.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)
    informative = np.where(y == 1, signal, -signal) + rng.standard_normal(n_samples)
    noise = rng.standard_normal((n_samples, n_noise)) * noise_scale
    X = np.column_stack([informative, noise])
    return X, y.astype(np.int64)


def make_count_data(
    n_samples: int = 2000,
    n_features: int = 20,
    seed: int = 42,
) -> tuple[Array, Labels]:
    """Two-class non-negative count data for Multinomial/Bernoulli Naive Bayes.

    Each class draws word counts from its own Poisson rates, so some "words" are
    class-indicative. Useful as a fast stand-in for bag-of-words text.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)

    base = rng.uniform(0.2, 1.5, size=n_features)
    bump = rng.uniform(0.5, 2.0, size=n_features) * (rng.random(n_features) < 0.4)
    rates = np.where(y[:, None] == 1, base + bump, base)
    X = rng.poisson(rates).astype(np.float64)
    return X, y.astype(np.int64)
