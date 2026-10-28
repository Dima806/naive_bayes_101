# naive_bayes_101

### The "wrong" model that refuses to lose

Naive Bayes rests on an assumption — that every feature is independent of every other,
given the class — that is **provably false on almost every real dataset**. By the rules of
statistical modeling it should be broken. Instead it classifies spam and text about as well
as far heavier methods, **trains in a single counting pass** with no optimization loop, and
**beats logistic regression when data is scarce**.

The catch: its predicted probabilities are **absurdly overconfident**, pinned near 0 and 1.
That is the whole lesson —

> **Getting the class right and getting the probability right are two different things.**
> Naive Bayes gets the decision right for the wrong reasons. Trust its decisions, never its probabilities.

This is a from-scratch, CPU-only, educational "101" project: three Naive Bayes variants built
on numpy, validated against scikit-learn to `1e-9`, wrapped in six notebooks, a Streamlit app,
and a small library.

---

## Quickstart

Runs inside a 2-CPU / 8 GB GitHub Codespace, no GPU. Requires [`uv`](https://docs.astral.sh/uv/).

```bash
make setup      # install deps + register the Jupyter kernel
make data       # cache the spam dataset (UCI SMS Spam, with a frozen fallback)
make test       # from-scratch models vs sklearn, smoothing, overconfidence (8 tests)
make notebooks  # run all 6 notebooks in place -> figures + JSON in outputs/
make run        # launch the Streamlit app
make ci         # sync + lint + test (what CI runs)
```

`make help` lists every target.

---

## What's inside

### The notebooks (`notebooks/`)

Each runs end to end in well under a minute and leans on the `src/` library.

| # | Notebook | What it shows |
|---|---|---|
| 01 | **Bayes to Classifier** | The classifier falls straight out of Bayes' theorem; the independence assumption named and flagged false; a worked example by hand |
| 02 | **Naive Bayes Is Counting** | Training is just means/variances or word counts — a single pass; the Gaussian decision boundary; why smoothing is mandatory |
| 03 | **The Spam Filter** | Multinomial NB on real SMS, matching logistic regression and training ~5× faster; the top spam words read straight off the counts |
| 04 | **Right Decision, Wrong Probability** *(showpiece)* | Probabilities pinned at the extremes; the calibration curve; a duplicated feature inflating confidence while accuracy holds |
| 05 | **When It Wins and Loses** | The data-efficiency crossover; high-dimensional competitiveness; the XOR interaction failure |
| 06 | **Practical Guide** | A decision table, all three variants working, and isotonic recalibration repairing the probabilities |

### The library (`src/`)

- **`models/`** — `GaussianNB`, `MultinomialNB`, `BernoulliNB` built from scratch (numpy/scipy only), plus `smoothing` (Laplace). Work in log space; matched to sklearn to `1e-9`.
- **`datasets/`** — `text` (bag-of-words spam, cached), `tabular`, and `synthetic` with two knobs: **feature correlation** and **training-set size**.
- **`evaluation/`** — decision quality, probability quality, data efficiency, the correlation effect, and the arena harness.
- **`visualisation.py`** / **`artifacts.py`** — plotting helpers, and `save_fig`/`save_json` so every figure lands in `outputs/figures/*.png` and every result in `outputs/data/*.json`.

### Outputs (`outputs/`)

Running the notebooks regenerates **10 figures** (`outputs/figures/*.png`) and **17 result files**
(`outputs/data/*.json`), each prefixed by notebook number.

---

## Headline results

All reproducible from the cached data and seeds in `config/settings.yaml`.

| Claim | Result |
|---|---|
| From-scratch Gaussian/Multinomial/Bernoulli vs sklearn | Match to **`1e-9`** (asserted in tests) |
| Spam: Multinomial NB vs logistic regression | **98.6%** accuracy, trains **~5× faster** |
| Overconfidence from a duplicated feature | Confidence **0.80 → 0.99** while accuracy stays **~0.79** |
| Data-efficiency crossover | NB leads from n=20; logistic overtakes around **n ≈ 640** |
| Unseen feature, no smoothing | Prediction **annihilated** (probability 0); smoothing fixes it |
| Isotonic recalibration | Log loss **2.55 → 0.53** with **identical** decisions |

---

## Related projects

Part of a wider set of hands-on data-science projects by [**@Dima806**](https://github.com/Dima806).
These connect directly to the ideas here:

| Project | Why it's relevant |
|---|---|
| [bayesian_stats_101](https://github.com/Dima806/bayesian_stats_101) | Bayes' theorem and Bayesian thinking — the engine underneath this classifier |
| [metrics_101](https://github.com/Dima806/metrics_101) | Precision, recall, F1, ROC/PR AUC, log loss — why good decisions and good probabilities are different numbers |
| [calibration_arena](https://github.com/Dima806/calibration_arena) | Platt, isotonic, temperature scaling and conformal prediction — how to repair the overconfident probabilities this project exposes |
| [embeddings_101](https://github.com/Dima806/embeddings_101) | From word co-occurrence counts to word2vec and sentence encoders — the richer cousin of the bag-of-words features used here |
| [text_cleaning_arena](https://github.com/Dima806/text_cleaning_arena) | Does the standard text-preprocessing pipeline actually help? The step before bag-of-words |
| [recommender_101](https://github.com/Dima806/recommender_101) | How recommendation really works — home of the popularity baseline, the recommender cousin of the strong baseline Naive Bayes provides for text |

---

## License

[Apache-2.0](LICENSE).
