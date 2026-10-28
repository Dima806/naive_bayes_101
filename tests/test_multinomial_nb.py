"""Multinomial Naive Bayes from scratch must match sklearn on text counts to 1e-9."""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB

from src.datasets import synthetic, text
from src.models import MultinomialNB

_CORPUS = [
    "free money win cash now",
    "win free prize claim now",
    "cheap loan offer click now",
    "lunch meeting tomorrow at noon",
    "project report due tomorrow",
    "see you at dinner tonight",
]
_LABELS = np.array([1, 1, 1, 0, 0, 0])


def test_multinomial_nb_matches_sklearn_on_text() -> None:
    X, _ = text.vectorize(_CORPUS, max_features=50)

    mine = MultinomialNB(alpha=1.0).fit(X, _LABELS)
    sk = SKMultinomialNB(alpha=1.0).fit(X, _LABELS)

    np.testing.assert_allclose(mine.feature_log_prob_, sk.feature_log_prob_, atol=1e-12)
    np.testing.assert_allclose(mine.class_log_prior_, sk.class_log_prior_, atol=1e-12)
    np.testing.assert_array_equal(mine.predict(X), sk.predict(X))
    np.testing.assert_allclose(mine.predict_proba(X), sk.predict_proba(X), atol=1e-9)


def test_multinomial_nb_matches_sklearn_on_counts() -> None:
    X, y = synthetic.make_count_data(n_samples=400, n_features=15, seed=0)

    mine = MultinomialNB(alpha=0.5).fit(X, y)
    sk = SKMultinomialNB(alpha=0.5).fit(X, y)

    np.testing.assert_allclose(mine.predict_proba(X), sk.predict_proba(X), atol=1e-9)
    np.testing.assert_array_equal(mine.predict(X), sk.predict(X))
