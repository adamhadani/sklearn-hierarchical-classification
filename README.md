# sklearn-hierarchical-classification

[![CI](https://github.com/promptromp/sklearn-hierarchical-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/promptromp/sklearn-hierarchical-classification/actions/workflows/ci.yml)
[![Docs](https://github.com/promptromp/sklearn-hierarchical-classification/actions/workflows/publish-docs.yml/badge.svg)](https://promptromp.github.io/sklearn-hierarchical-classification/)
[![PyPI - Version](https://img.shields.io/pypi/v/sklearn-hierarchical-classification)](https://pypi.org/project/sklearn-hierarchical-classification/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sklearn-hierarchical-classification)](https://pypi.org/project/sklearn-hierarchical-classification/)
[![License](https://img.shields.io/github/license/promptromp/sklearn-hierarchical-classification)](./LICENSE)

Hierarchical classification for [scikit-learn](https://scikit-learn.org/).

`HierarchicalClassifier` is a scikit-learn compatible meta-estimator that trains a local
classifier at every parent node of a class hierarchy, given as a tree or DAG (a
`networkx.DiGraph` or an adjacency dict), and predicts by walking the hierarchy top-down. It
supports any base classifier (one per node if needed, or a `Pipeline` over raw inputs such as
text), early stopping at intermediate nodes, and multi-label targets through a
`MultiLabelBinarizer`, with per-class decision thresholds, an "inclusive" training strategy that
lets a node reject what its parent mis-routes to it, and a `thresholds` module that tunes
thresholds from held-out scores. Hierarchical precision, recall and F-beta are in `metrics`.

**Documentation:** <https://promptromp.github.io/sklearn-hierarchical-classification/>

> **Origins.** This project was created at Globality and lived at
> [globality-corp/sklearn-hierarchical-classification](https://github.com/globality-corp/sklearn-hierarchical-classification)
> through release 1.3.2, after which that repository was archived. This repository is its
> continuation under the same package name on PyPI, maintained by the original author: the
> estimator was brought up to current scikit-learn, made faster, and extended with multi-label
> threshold tuning and benchmarks. Coming from 1.3.x? See the
> [upgrade notes](https://promptromp.github.io/sklearn-hierarchical-classification/upgrading).


## Installation

Requires Python 3.11+.

    pip install sklearn-hierarchical-classification

Or with [uv](https://docs.astral.sh/uv/):

    uv add sklearn-hierarchical-classification


## Usage

```python
from sklearn.linear_model import LogisticRegression

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT

# A two-level hierarchy over the digits 1, 3, 7, 8, 9. Intermediate nodes ("A", "B") are
# arbitrary labels; the (artificial) root is the framework-provided ROOT constant.
class_hierarchy = {
    ROOT: ["A", "B"],
    "A": ["1", "7"],
    "B": ["3", "8", "9"],
}

clf = HierarchicalClassifier(
    base_estimator=LogisticRegression(),
    class_hierarchy=class_hierarchy,
)
clf.fit(X_train, y_train)  # y_train holds leaf labels, e.g. "1", "7", ...
y_pred = clf.predict(X_test)
```

The base estimator must expose `predict_proba` (or `decision_function`, with
`use_decision_function=True`). For multi-label hierarchies pass the fitted `MultiLabelBinarizer`
as `mlb`:

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

clf = HierarchicalClassifier(
    base_estimator=OneVsRestClassifier(LinearSVC()),
    class_hierarchy=class_hierarchy,
    mlb=mlb,                          # fitted on the label sets; fit() takes mlb.transform(labels)
    use_decision_function=True,
    training_strategy="inclusive",    # out-of-subtree samples are negatives at every node
    mlb_prediction_threshold=0.0,     # one threshold, or one per class from the thresholds module
    mlb_min_root_predictions=1,       # no sample is left without a top-level label
)
```

The [documentation](https://promptromp.github.io/sklearn-hierarchical-classification/) covers
hierarchies and base estimators, raw-text pipelines, early stopping, multi-label prediction and
threshold tuning, the hierarchical metrics, and the API. `examples/classify_digits.py` is a
complete, self-contained example including the metrics.


## Benchmarks

Both scripts in `benchmarks/` choose every tuned setting on out-of-fold or development data and
score the test set once. Full tables and protocol are on the
[Benchmarks](https://promptromp.github.io/sklearn-hierarchical-classification/benchmarks) page.

**RCV1-v2** (Lewis et al. 2004: 103 topics, 23,149 training and 781,265 test newswire
documents, TF-IDF features from scikit-learn), `LinearSVC` local classifiers:

| Model | micro-F1 | macro-F1 |
|---|---:|---:|
| Flat `OneVsRest(LinearSVC)` | 0.804 | 0.486 |
| Hierarchical, siblings-trained nodes | 0.796 | 0.514 |
| Hierarchical, inclusive-trained nodes + routed per-class thresholds from CV (`--tune --C 0.5`) | **0.816** | **0.609** |
| Published SVM with per-category tuned thresholds | 0.816 | 0.607 |

**GermEval 2019 Task 1** (German book blurbs, 343 genres in a 4-level tree), test set, with the
feature set, threshold and root fallback chosen on the development split:

| Configuration | micro-F1 | macro-F1 | root genres micro-F1 |
|---|---:|---:|---:|
| Title + blurb TF-IDF, dev-selected threshold and root fallback | 0.651 | 0.282 | 0.807 |
| Plus metadata views (title, author tokens, ISBN publisher prefixes) | **0.725** | **0.373** | **0.872** |
| Published winning system (built on this library, per-node vocabularies) | 0.677 | | 0.863 |

**Blurb Genre Collection** (English book blurbs from the same group, 146 genres in a 4-level
hierarchy, 58,715 / 14,785 / 18,394 train / dev / test), one of the four standard datasets of
the hierarchical text classification literature, same protocol. The neural rows use the blurb text
only, so the text-only row is the like-for-like comparison; the metadata row shows what the
dataset's own fields add. Classifier fit takes 3.5 minutes on a laptop core against hours on a GPU:

| Configuration | micro-F1 | macro-F1 |
|---|---:|---:|
| Title + blurb TF-IDF, dev-selected threshold and root fallback | 0.768 | 0.552 |
| Plus metadata views (title, author tokens, ISBN publisher prefixes) | 0.818 | 0.622 |
| SVM baseline of the dataset paper (Aly et al. 2019) | 0.712 | |
| Fine-tuned BERT-base, flat (Karl and Scherp 2025) | 0.814 | 0.646 |
| HYDRA, RoBERTa-base with per-level heads (EMNLP 2025) | **0.822** | **0.662** |

    uv run python benchmarks/rcv1_benchmark.py --tune --C 0.5
    uv run python benchmarks/germeval2019_benchmark.py
    uv run python benchmarks/bgc_benchmark.py


## Development

The project is managed with [uv](https://docs.astral.sh/uv/); all tooling configuration lives in
`pyproject.toml` and is enforced through [pre-commit](https://pre-commit.com/) locally and in CI.

    uv sync --dev                       # create .venv and install the package + dev tools
    uv run pre-commit install           # run the hooks on every commit
    uv run pytest -m "not slow"         # what the pre-commit hook runs
    uv run pre-commit run --all-files   # everything CI enforces: ruff, codespell, mypy, hygiene hooks, tests

Versions come from git tags (setuptools-scm); pushing a tag `X.Y.Z` publishes to PyPI and
creates a GitHub Release. The documentation site is built by Jekyll from `docs/` and published
to GitHub Pages on every push to `develop` that touches it. See [CONTRIBUTING.md](./CONTRIBUTING.md) and the
[Development](https://promptromp.github.io/sklearn-hierarchical-classification/development) page.

Upgrading from the 1.3.x releases? Read the
[upgrade notes](https://promptromp.github.io/sklearn-hierarchical-classification/upgrading).


## Credits

Created by Globality Engineering and released under the Apache License 2.0 (see the Origins
note above); maintained under the [PromptRomp](https://github.com/promptromp) organization.

The design follows the framework of
["A survey of hierarchical classification across different application domains"](https://www.researchgate.net/publication/225716424_A_survey_of_hierarchical_classification_across_different_application_domains)
(Silla and Freitas 2011), and the hierarchical metrics
["Functional Annotation of Genes Using Hierarchical Text Categorization"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf)
(Kiritchenko et al. 2005). Further reading:

* ["Classifying web documents in a hierarchy of categories: a comprehensive study"](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.150.8859) - Ceci and Malerba 2007
* ["A Survey of Automated Hierarchical Classification of Patents"](https://lirias.kuleuven.be/bitstream/123456789/457904/1/GomezMoens%20Mumia_book_chapter_camera_ready2014.pdf) - Gomez et al. 2014
* ["Evaluation Measures for Hierarchical Classification: a unified view and novel approaches"](https://arxiv.org/pdf/1306.6802.pdf) - Kosmopoulos et al. 2013
* ["Bayesian Aggregation for Hierarchical Classification"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.3312&rep=rep1&type=pdf) - Barutcuoglu et al. 2008
* ["Kaggle LSHTC4 Winning Solution"](https://kaggle2.blob.core.windows.net/forum-message-attachments/43550/1230/lshtc4.pdf) - Puurula et al. 2014
* ["Feature-Weighted Linear Stacking"](https://arxiv.org/pdf/0911.0460.pdf) - Sill et al. 2009
