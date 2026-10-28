"""Persist notebook outputs so the app never need to re-run anything.

Every figure is written to ``outputs/figures/<name>.png`` and every textual or numerical
result to ``outputs/data/<name>.json``. Notebooks prefix names with their number
(e.g. ``03_top_spam_words``) so artifacts stay grouped and discoverable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DATA_DIR = PROJECT_ROOT / "outputs" / "data"


def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays JSON-serialisable."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def save_fig(fig: Figure, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure as ``outputs/figures/<name>.png`` and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_json(data: Any, name: str) -> Path:
    """Save text/numbers as ``outputs/data/<name>.json`` (numpy-aware) and return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.json"
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, default=_json_default)
    return path
