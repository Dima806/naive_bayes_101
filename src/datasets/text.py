"""Spam/ham text with bag-of-words features — Naive Bayes' home turf.

Offline-first and reproducible:

1. If a cached CSV exists in ``text.cache_dir`` it is used.
2. Otherwise a best-effort download of a permissively-licensed spam set is attempted.
3. On any failure, a **frozen synthetic** spam corpus is generated from fixed word
   distributions and cached, so ``make data`` and the notebooks always work.

Run ``python -m src.datasets.text`` (``make data``) to populate the cache.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_extraction.text import CountVectorizer

from src.config import PROJECT_ROOT, SETTINGS

Array = NDArray[np.float64]
Labels = NDArray[np.int64]

_SMS_SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
_SMS_MEMBER = "SMSSpamCollection"

# Fixed vocabularies for the frozen synthetic fallback. Deliberately overlapping so the
# problem is non-trivial and the model has to weigh evidence rather than memorise words.
_SPAM_WORDS = [
    "free",
    "win",
    "winner",
    "cash",
    "prize",
    "claim",
    "urgent",
    "offer",
    "now",
    "click",
    "limited",
    "money",
    "guaranteed",
    "credit",
    "loan",
    "congratulations",
    "call",
    "txt",
    "reply",
    "stop",
    "subscribe",
    "deal",
]
_HAM_WORDS = [
    "meeting",
    "lunch",
    "tomorrow",
    "project",
    "email",
    "send",
    "please",
    "thanks",
    "see",
    "you",
    "later",
    "home",
    "dinner",
    "call",
    "back",
    "soon",
    "work",
    "report",
    "schedule",
    "happy",
    "weekend",
    "family",
]
_SHARED_WORDS = [
    "the",
    "a",
    "to",
    "you",
    "your",
    "is",
    "are",
    "and",
    "for",
    "of",
    "on",
    "now",
    "call",
    "please",
]


def _cache_dir() -> Path:
    path = PROJECT_ROOT / SETTINGS.text.cache_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_synthetic_corpus(n_samples: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Build the frozen synthetic spam/ham corpus (deterministic given the seed)."""
    n_samples = n_samples if n_samples is not None else SETTINGS.text.n_samples
    seed = seed if seed is not None else SETTINGS.seed
    rng = np.random.default_rng(seed)

    rows: list[dict[str, object]] = []
    for _ in range(n_samples):
        is_spam = bool(rng.random() < 0.4)
        topic = _SPAM_WORDS if is_spam else _HAM_WORDS
        length = int(rng.integers(6, 18))
        words = []
        for _ in range(length):
            pool = topic if rng.random() < 0.7 else _SHARED_WORDS
            words.append(str(rng.choice(pool)))
        rows.append({"text": " ".join(words), "label": int(is_spam)})
    return pd.DataFrame(rows)


def _try_download() -> pd.DataFrame | None:
    """Best-effort fetch of the UCI SMS Spam Collection. Returns None on any failure."""
    try:
        with urlopen(_SMS_SPAM_URL, timeout=10) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            raw = archive.read(_SMS_MEMBER).decode("latin-1")
        records = [line.split("\t", 1) for line in raw.splitlines() if "\t" in line]
        frame = pd.DataFrame(records, columns=["label_str", "text"])
        frame["label"] = (frame["label_str"] == "spam").astype(int)
        return frame[["text", "label"]]
    except Exception:
        return None


def load_text_data(allow_download: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame with ``text`` and integer ``label`` (1 = spam, 0 = ham)."""
    cache_file = _cache_dir() / "messages.csv"
    if cache_file.exists() and not refresh:
        return pd.read_csv(cache_file)

    frame = _try_download() if allow_download else None
    source = "download"
    if frame is None:
        frame = make_synthetic_corpus()
        source = "frozen-synthetic"

    frame.to_csv(cache_file, index=False)
    frame.attrs["source"] = source
    return frame


def vectorize(texts: list[str], max_features: int | None = None) -> tuple[Array, CountVectorizer]:
    """Bag-of-words count matrix (dense) and the fitted vectorizer."""
    max_features = max_features if max_features is not None else SETTINGS.text.max_features
    vectorizer = CountVectorizer(max_features=max_features)
    counts = vectorizer.fit_transform(texts)
    return counts.toarray().astype(np.float64), vectorizer


def main() -> None:
    frame = load_text_data(refresh=True)
    spam = int(frame["label"].sum())
    print(f"✅ Cached {len(frame)} messages ({spam} spam / {len(frame) - spam} ham)")
    print(f"   -> {_cache_dir() / 'messages.csv'}")


if __name__ == "__main__":
    main()
