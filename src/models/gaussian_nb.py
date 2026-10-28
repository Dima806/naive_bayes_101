"""Gaussian Naive Bayes from scratch — for continuous features.

Model each feature, per class, as a 1-D Gaussian: estimate its mean and variance by
*counting* (averaging). "Training" is nothing but computing per-class means and
variances in a single pass. Validated against ``sklearn.naive_bayes.GaussianNB`` to 1e-9.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import BaseNB

Array = NDArray[np.float64]


class GaussianNB(BaseNB):
    """Gaussian Naive Bayes with the same ``var_smoothing`` stabiliser as scikit-learn."""

    theta_: Array  # per-class feature means, shape (n_classes, n_features)
    var_: Array  # per-class feature variances (with epsilon added)
    class_count_: Array
    class_prior_: Array
    epsilon_: float

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X: Array, y: NDArray) -> GaussianNB:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]

        # Portion of the largest feature variance added for numerical stability.
        self.epsilon_ = self.var_smoothing * float(np.var(X, axis=0).max(axis=0))

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_count_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i, :] = X_c.mean(axis=0)
            self.var_[i, :] = X_c.var(axis=0)  # population variance (ddof=0), like sklearn
            self.class_count_[i] = X_c.shape[0]
        self.var_ += self.epsilon_
        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        return self

    def _joint_log_likelihood(self, X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        joint = []
        for i in range(self.classes_.shape[0]):
            prior = np.log(self.class_prior_[i])
            likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i, :]))
            likelihood -= 0.5 * np.sum((X - self.theta_[i, :]) ** 2 / self.var_[i, :], axis=1)
            joint.append(prior + likelihood)
        return np.array(joint).T
