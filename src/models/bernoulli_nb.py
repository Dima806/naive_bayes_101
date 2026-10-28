"""Bernoulli Naive Bayes from scratch — for binary presence/absence features.

Models *whether* each feature appears, not how often. Unlike Multinomial NB, it
explicitly penalises features that are absent from a document. Included to complete
the family and to show that the model form should match the feature type. Validated
against ``sklearn.naive_bayes.BernoulliNB`` to 1e-9.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import smoothing
from .base import BaseNB

Array = NDArray[np.float64]


class BernoulliNB(BaseNB):
    """Bernoulli Naive Bayes; inputs are binarised at ``binarize`` before counting."""

    feature_count_: Array
    class_count_: Array
    feature_log_prob_: Array  # log P(feature present | class)
    class_log_prior_: Array

    def __init__(
        self, alpha: float = 1.0, binarize: float | None = 0.0, fit_prior: bool = True
    ) -> None:
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior

    def _binarize(self, X: Array) -> Array:
        if self.binarize is None:
            return X
        return (self.binarize < X).astype(np.float64)

    def fit(self, X: Array, y: NDArray) -> BernoulliNB:
        X = self._binarize(np.asarray(X, dtype=np.float64))
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

        self.feature_log_prob_ = smoothing.bernoulli_feature_log_prob(
            self.feature_count_, self.class_count_, self.alpha
        )
        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        return self

    def _joint_log_likelihood(self, X: Array) -> Array:
        X = self._binarize(np.asarray(X, dtype=np.float64))
        neg_prob = np.log(1.0 - np.exp(self.feature_log_prob_))
        jll = X @ (self.feature_log_prob_ - neg_prob).T
        jll += self.class_log_prior_ + neg_prob.sum(axis=1)
        return jll
