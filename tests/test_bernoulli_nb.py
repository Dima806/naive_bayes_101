"""Bernoulli Naive Bayes from scratch must match sklearn to 1e-9."""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import BernoulliNB as SKBernoulliNB

from src.datasets import synthetic
from src.models import BernoulliNB


def test_bernoulli_nb_matches_sklearn() -> None:
    X, y = synthetic.make_count_data(n_samples=400, n_features=15, seed=1)

    mine = BernoulliNB(alpha=1.0, binarize=0.0).fit(X, y)
    sk = SKBernoulliNB(alpha=1.0, binarize=0.0).fit(X, y)

    np.testing.assert_allclose(mine.feature_log_prob_, sk.feature_log_prob_, atol=1e-12)
    np.testing.assert_array_equal(mine.predict(X), sk.predict(X))
    np.testing.assert_allclose(mine.predict_proba(X), sk.predict_proba(X), atol=1e-9)
