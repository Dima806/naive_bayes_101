"""Gaussian Naive Bayes from scratch must match sklearn to 1e-9."""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import GaussianNB as SKGaussianNB

from src.datasets import tabular
from src.models import GaussianNB


def test_gaussian_nb_matches_sklearn() -> None:
    X, y, _ = tabular.load_continuous()

    mine = GaussianNB().fit(X, y)
    sk = SKGaussianNB().fit(X, y)

    # Learned parameters are computed identically.
    np.testing.assert_allclose(mine.theta_, sk.theta_, atol=1e-12)
    np.testing.assert_allclose(mine.var_, sk.var_, atol=1e-12)

    # Predictions and probabilities match to 1e-9.
    np.testing.assert_array_equal(mine.predict(X), sk.predict(X))
    np.testing.assert_allclose(mine.predict_proba(X), sk.predict_proba(X), atol=1e-9)
    np.testing.assert_allclose(
        mine._joint_log_likelihood(X), sk._joint_log_likelihood(X), atol=1e-6
    )


def test_gaussian_nb_proba_rows_sum_to_one() -> None:
    X, y, _ = tabular.load_continuous()
    proba = GaussianNB().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-12)
