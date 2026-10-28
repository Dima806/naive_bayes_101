"""Shared machinery for the Naive Bayes classifiers.

Every variant differs only in how it computes the *joint log-likelihood*
``log P(class) + sum_j log P(feature_j | class)``. Prediction, probability
normalisation and the argmax decision rule are identical, and live here.

This mirrors the structure of scikit-learn's ``_BaseNB`` so the from-scratch
models can be validated against it to 1e-9.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


class BaseNB:
    """Base class: subclasses set ``classes_`` and implement the joint log-likelihood."""

    classes_: NDArray[np.int64]

    def _joint_log_likelihood(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the unnormalised ``log P(class, X)``, shape ``(n_samples, n_classes)``."""
        raise NotImplementedError

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Pick the class with the highest joint log-likelihood (the decision rule)."""
        jll = self._joint_log_likelihood(np.asarray(X, dtype=np.float64))
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalise the joint log-likelihood into ``log P(class | X)``."""
        jll = self._joint_log_likelihood(np.asarray(X, dtype=np.float64))
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``P(class | X)``. Correct for ranking, badly miscalibrated as numbers."""
        return np.exp(self.predict_log_proba(X))
