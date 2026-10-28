"""Shared plotting helpers (matplotlib) for the notebooks figures.

Kept deliberately small: each function takes already-computed arrays and returns a
Figure, so the notebooks stay thin and the evaluation logic stays in ``src/evaluation``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .models.base import BaseNB

Array = NDArray[np.float64]


def plot_decision_boundary(model: BaseNB, X: Array, y: NDArray, resolution: int = 200) -> Figure:
    """Draw a trained 2-feature classifier's decision regions with the data on top."""
    if X.shape[1] != 2:
        raise ValueError("decision boundary plot needs exactly 2 features")

    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    x_min, x_max = float(col_min[0]) - 1.0, float(col_max[0]) + 1.0
    y_min, y_max = float(col_min[1]) - 1.0, float(col_max[1]) + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, zz, alpha=0.2, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
    ax.set_title("Gaussian Naive Bayes decision boundary")
    return fig


def plot_calibration(mean_predicted: Array, observed: Array) -> Figure:
    """Reliability diagram: the diagonal is perfect calibration; Naive Bayes hugs the corners."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot(mean_predicted, observed, "o-", label="Naive Bayes")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration curve")
    ax.legend()
    return fig


def plot_data_efficiency(sizes: list[int], series: dict[str, list[float]]) -> Figure:
    """Accuracy versus training-set size for one or more models (the crossover plot)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, accuracies in series.items():
        ax.plot(sizes, accuracies, "o-", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("training-set size")
    ax.set_ylabel("test accuracy")
    ax.set_title("Data efficiency: who wins when data is scarce?")
    ax.legend()
    return fig


def plot_overconfidence_vs_correlation(
    correlations: list[float], confidences: list[float], accuracies: list[float]
) -> Figure:
    """Confidence climbing with correlation while accuracy stays flat — the paradox, plotted."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(correlations, confidences, "o-", label="mean confidence")
    ax.plot(correlations, accuracies, "s--", label="accuracy")
    ax.set_xlabel("feature correlation")
    ax.set_ylabel("value")
    ax.set_title("Overconfidence grows with correlation; accuracy does not")
    ax.legend()
    return fig


def plot_top_words(words: list[str], weights: list[float], title: str = "Top words") -> Figure:
    """Horizontal bar chart of the words that most move a message toward a class."""
    fig, ax = plt.subplots(figsize=(6, 5))
    order = np.argsort(weights)
    sorted_words = [words[i] for i in order]
    sorted_weights = [weights[i] for i in order]
    ax.barh(sorted_words, sorted_weights, color="tab:red")
    ax.set_title(title)
    return fig
