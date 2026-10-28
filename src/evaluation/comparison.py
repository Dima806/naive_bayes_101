"""The arena harness — line up Naive Bayes against logistic regression on one split.

Reports decision quality (accuracy, F1, train time) and probability quality
(mean confidence, log loss) side by side, so the "right decision, wrong probability"
story reads straight off the table.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .decision import evaluate_decision
from .probability import log_loss_score, mean_confidence

Array = NDArray[np.float64]


class ScoringClassifier(Protocol):
    """A classifier that also reports probabilities — required for the arena."""

    def fit(self, X: Array, y: NDArray) -> object: ...
    def predict(self, X: Array) -> NDArray: ...
    def predict_proba(self, X: Array) -> NDArray: ...


def run_arena(
    models: dict[str, ScoringClassifier],
    X_train: Array,
    y_train: NDArray,
    X_test: Array,
    y_test: NDArray,
) -> pd.DataFrame:
    """Evaluate each named model and return a tidy comparison table."""
    rows: list[dict[str, object]] = []
    for name, model in models.items():
        decision = evaluate_decision(model, X_train, y_train, X_test, y_test)
        proba = np.asarray(model.predict_proba(X_test))

        rows.append(
            {
                "model": name,
                "accuracy": decision.accuracy,
                "f1": decision.f1,
                "train_seconds": decision.train_seconds,
                "mean_confidence": mean_confidence(proba),
                "log_loss": log_loss_score(y_test, proba),
            }
        )
    return pd.DataFrame(rows).set_index("model")
