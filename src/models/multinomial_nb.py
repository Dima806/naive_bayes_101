"""Multinomial Naive Bayes from scratch — the text workhorse.

For count features (e.g. bag-of-words), estimate ``P(word | class)`` by literally
counting how often each word occurs in each class. "Training" is summing word counts
per class in a single pass. This is the engine of the spam filter. Validated against
``sklearn.naive_bayes.MultinomialNB`` to 1e-9.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import smoothing
from .base import BaseNB

Array = NDArray[np.float64]


class MultinomialNB(BaseNB):
    """Multinomial Naive Bayes with Laplace/additive smoothing (``alpha``)."""

    feature_count_: Array  # summed counts per class, shape (n_classes, n_features)
    class_count_: Array
    feature_log_prob_: Array  # log P(feature | class)
    class_log_prior_: Array

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True) -> None:
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X: Array, y: NDArray) -> MultinomialNB:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]

        self.feature_count_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.feature_count_[i, :] = X_c.sum(axis=0)
            self.class_count_[i] = X_c.shape[0]

        self.feature_log_prob_ = smoothing.multinomial_feature_log_prob(
            self.feature_count_, self.alpha
        )
        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        return self

    def _joint_log_likelihood(self, X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.feature_log_prob_.T + self.class_log_prior_
