"""Data efficiency — accuracy versus training-set size, and the crossover.

Naive Bayes wins when data is scarce (it cannot overfit); logistic regression overtakes
as data grows. This sweeps training-set size and records test accuracy for each model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .decision import Classifier

Array = NDArray[np.float64]
ModelFactory = Callable[[], Classifier]


def accuracy_vs_size(
    model_factory: ModelFactory,
    X_train: Array,
    y_train: NDArray,
    X_test: Array,
    y_test: NDArray,
    sizes: list[int],
    seed: int = 42,
) -> tuple[list[int], list[float]]:
    """Train on the first ``n`` shuffled rows for each ``n`` in ``sizes``; return accuracies."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_train))
    X_shuffled, y_shuffled = X_train[order], np.asarray(y_train)[order]

    used: list[int] = []
    accuracies: list[float] = []
    for n in sizes:
        if n > len(y_shuffled) or len(np.unique(y_shuffled[:n])) < 2:
            continue
        model = model_factory()
        model.fit(X_shuffled[:n], y_shuffled[:n])
        preds = model.predict(X_test)
        accuracies.append(float((preds == y_test).mean()))
        used.append(n)
    return used, accuracies
