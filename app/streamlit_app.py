"""Naive Bayes 101 — interactive demos.

Four tabs:
1. Spam classifier — type a message, see the score and the words pushing it toward spam.
2. Counting viewer — the per-feature counts that *are* the trained model.
3. Overconfidence demo — duplicate a feature, watch confidence march toward 1.0.
4. Data-efficiency race — shrink the training set, watch Naive Bayes overtake logistic.

Run with ``make run``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import SETTINGS
from src.datasets import synthetic, text
from src.evaluation.data_efficiency import accuracy_vs_size
from src.evaluation.probability import mean_confidence
from src.models import GaussianNB, MultinomialNB

st.set_page_config(page_title="Naive Bayes 101", layout="wide")
st.title("Naive Bayes 101 — the wrong model that refuses to lose")


@st.cache_data
def _load_spam_model() -> tuple[MultinomialNB, object, pd.DataFrame]:
    frame = text.load_text_data()
    X, vectorizer = text.vectorize(frame["text"].tolist(), max_features=SETTINGS.text.max_features)
    model = MultinomialNB(alpha=1.0).fit(X, frame["label"].to_numpy())
    return model, vectorizer, frame


spam_tab, counts_tab, overconf_tab, race_tab = st.tabs(
    ["Spam classifier", "Counting viewer", "Overconfidence", "Data-efficiency race"]
)

with spam_tab:
    st.header("Spam classifier")
    model, vectorizer, _ = _load_spam_model()
    message = st.text_area("Message", "win a free prize now, click to claim your cash")
    if message.strip():
        x = vectorizer.transform([message]).toarray().astype(float)
        proba = model.predict_proba(x)[0]
        spam_index = int(np.where(model.classes_ == 1)[0][0])
        st.metric("P(spam)", f"{proba[spam_index]:.6f}")
        st.caption("Note how extreme the probability is — right decision, wrong probability.")

        vocab = np.array(vectorizer.get_feature_names_out())
        weights = model.feature_log_prob_[spam_index] - model.feature_log_prob_[1 - spam_index]
        present = x[0] > 0
        if present.any():
            pushes = pd.DataFrame(
                {"word": vocab[present], "spam_push": weights[present]}
            ).sort_values("spam_push", ascending=False)
            st.subheader("Words pushing this message toward spam")
            st.dataframe(pushes, use_container_width=True)

with counts_tab:
    st.header("Counting viewer — the model is just tallies")
    model, vectorizer, _ = _load_spam_model()
    vocab = np.array(vectorizer.get_feature_names_out())
    chosen = st.radio("Class", ["ham (0)", "spam (1)"], horizontal=True)
    class_index = 1 if chosen.startswith("spam") else 0
    counts = pd.DataFrame({"word": vocab, "count": model.feature_count_[class_index]}).sort_values(
        "count", ascending=False
    )
    st.dataframe(counts.head(30), use_container_width=True)

with overconf_tab:
    st.header("Overconfidence demo — duplicate a feature")
    n_copies = st.slider("Extra copies of the informative feature", 0, 20, 0)
    X, y = synthetic.make_informative_plus_noise(n_samples=800, n_noise=3, signal=0.8, seed=0)
    X_demo = synthetic.duplicate_feature(X, 0, n_copies) if n_copies else X
    demo_model = GaussianNB().fit(X_demo, y)
    confidence = mean_confidence(demo_model.predict_proba(X_demo))
    accuracy = float((demo_model.predict(X_demo) == y).mean())
    col_conf, col_acc = st.columns(2)
    col_conf.metric("Mean confidence", f"{confidence:.4f}")
    col_acc.metric("Accuracy", f"{accuracy:.4f}")
    st.caption("Confidence climbs with every duplicate; accuracy barely moves.")

with race_tab:
    st.header("Data-efficiency race — Naive Bayes vs logistic regression")
    X, y = synthetic.make_correlated_data(
        n_samples=4000, n_features=20, correlation=0.0, class_sep=1.2, seed=1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SETTINGS.test_size, random_state=SETTINGS.seed
    )
    sizes = SETTINGS.synthetic.sample_sizes
    nb_sizes, nb_acc = accuracy_vs_size(GaussianNB, X_train, y_train, X_test, y_test, sizes)
    lr_sizes, lr_acc = accuracy_vs_size(
        lambda: LogisticRegression(max_iter=1000), X_train, y_train, X_test, y_test, sizes
    )
    chart = pd.DataFrame({"Naive Bayes": nb_acc}, index=nb_sizes)
    chart["Logistic Regression"] = pd.Series(dict(zip(lr_sizes, lr_acc, strict=False)))
    st.line_chart(chart)
    st.caption("Naive Bayes tends to lead when data is scarce; logistic catches up as it grows.")
