"""Smoothing is a requirement, not a refinement.

Without it, a feature value unseen in a class during training gets probability 0 and
annihilates the prediction. With Laplace smoothing, no probability is ever exactly 0.
"""

from __future__ import annotations

import numpy as np

from src.models import MultinomialNB

# Two "words". Word 1 never appears in class 0's training data.
X_TRAIN = np.array(
    [
        [1.0, 0.0],  # class 0: only word 0
        [2.0, 0.0],  # class 0: only word 0
        [0.0, 1.0],  # class 1: only word 1
        [0.0, 2.0],  # class 1: only word 1
    ]
)
Y_TRAIN = np.array([0, 0, 1, 1])

# A test document containing the word that class 0 never saw.
X_TEST = np.array([[1.0, 1.0]])


def test_unseen_feature_annihilates_without_smoothing() -> None:
    model = MultinomialNB(alpha=0.0).fit(X_TRAIN, Y_TRAIN)
    # P(word 1 | class 0) == 0  ->  log == -inf
    assert np.isneginf(model.feature_log_prob_[0, 1])
    # The joint log-likelihood for class 0 collapses to -inf on the test doc.
    jll = model._joint_log_likelihood(X_TEST)
    assert np.isneginf(jll[0, 0])


def test_smoothing_keeps_all_probabilities_positive() -> None:
    model = MultinomialNB(alpha=1.0).fit(X_TRAIN, Y_TRAIN)
    probs = np.exp(model.feature_log_prob_)
    assert np.all(probs > 0.0)
    assert np.all(np.isfinite(model.feature_log_prob_))

    proba = model.predict_proba(X_TEST)
    assert np.all(proba > 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
