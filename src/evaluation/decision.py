"""Decision quality — does the model pick the right class? (Accuracy / F1, and speed.)

This is the thing Naive Bayes is secretly good at, even when its probabilities are junk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score

Array = NDArray[np.float64]


class Classifier(Protocol):
    """Minimal fit/predict interface shared by the from-scratch models and sklearn."""

    def fit(self, X: Array, y: NDArray) -> object: ...
    def predict(self, X: Array) -> NDArray: ...


@dataclass
class DecisionResult:
    accuracy: float
    f1: float
    train_seconds: float


def evaluate_decision(
    model: Classifier,
    X_train: Array,
    y_train: NDArray,
    X_test: Array,
    y_test: NDArray,
) -> DecisionResult:
    """Fit on train, score decisions on test, and time the (single-pass) training."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    preds = model.predict(X_test)
    return DecisionResult(
        accuracy=float(accuracy_score(y_test, preds)),
        f1=float(f1_score(y_test, preds, average="macro")),
        train_seconds=train_seconds,
    )
