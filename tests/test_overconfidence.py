"""Duplicating a feature (perfect correlation) increases overconfidence, not accuracy.

This is the mechanical heart of the paradox: Naive Bayes double-counts the duplicated
evidence, so its confidence climbs while its decisions barely change.
"""

from __future__ import annotations

from src.datasets import synthetic
from src.evaluation.probability import mean_confidence
from src.models import GaussianNB


def _confidence_and_accuracy(X, y) -> tuple[float, float]:
    model = GaussianNB().fit(X, y)
    confidence = mean_confidence(model.predict_proba(X))
    accuracy = float((model.predict(X) == y).mean())
    return confidence, accuracy


def test_duplicated_feature_increases_overconfidence() -> None:
    # One informative feature (column 0) plus noise, so column 0 decides the class.
    X, y = synthetic.make_informative_plus_noise(n_samples=800, n_noise=3, signal=0.8, seed=0)
    base_conf, base_acc = _confidence_and_accuracy(X, y)

    # Duplicate the informative feature: the model double-counts the same evidence.
    X_dup = synthetic.duplicate_feature(X, feature_index=0, n_copies=6)
    dup_conf, dup_acc = _confidence_and_accuracy(X_dup, y)

    # Overconfidence climbs clearly...
    assert dup_conf > base_conf + 0.1
    # ...while accuracy barely moves.
    assert abs(dup_acc - base_acc) <= 0.05
