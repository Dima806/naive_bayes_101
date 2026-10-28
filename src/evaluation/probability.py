"""Probability quality — are the reported probabilities any good? (They are not.)

Naive Bayes pins probabilities near 0 and 1. These helpers quantify that overconfidence
(mean top-class confidence, log loss) and produce the calibration curve for notebook 04.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

Array = NDArray[np.float64]


def mean_confidence(proba: Array) -> float:
    """Average probability assigned to the predicted (top) class. Near 1.0 = overconfident."""
    return float(np.max(proba, axis=1).mean())


def log_loss_score(y_true: NDArray, proba: Array) -> float:
    """Log loss: punishes confident-and-wrong hard. Naive Bayes scores badly here."""
    return float(log_loss(y_true, proba))


def calibration_data(
    y_true: NDArray, proba_positive: Array, n_bins: int = 10
) -> tuple[Array, Array]:
    """Return ``(mean_predicted, observed_frequency)`` points for a reliability diagram."""
    frac_pos, mean_pred = calibration_curve(y_true, proba_positive, n_bins=n_bins)
    return np.asarray(mean_pred), np.asarray(frac_pos)
