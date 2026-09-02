# sklearn-hierarchical-classification

[![CI](https://github.com/adamhadani/sklearn-hierarchical-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/adamhadani/sklearn-hierarchical-classification/actions/workflows/ci.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/sklearn-hierarchical-classification)](https://pypi.org/project/sklearn-hierarchical-classification/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sklearn-hierarchical-classification)](https://pypi.org/project/sklearn-hierarchical-classification/)
[![License](https://img.shields.io/github/license/adamhadani/sklearn-hierarchical-classification)](./LICENSE)

Hierarchical classification module based on scikit-learn's interfaces and conventions.

`HierarchicalClassifier` is a scikit-learn compatible meta-estimator that fits a "local classifier per
parent node" over a class hierarchy given as a tree or DAG (a `networkx.DiGraph` or an adjacency dict),
and predicts by walking the hierarchy top-down. It supports mandatory and non-mandatory leaf-node
prediction (early stopping), per-node base estimators, and a "raw" feature-extraction mode where the
base estimator is a full `Pipeline` operating on raw inputs such as text. Multi-label targets are
supported in both modes via a `MultiLabelBinarizer` passed as `mlb`. Hierarchical precision/recall/F-beta
metrics are provided in `sklearn_hierarchical_classification.metrics`.

See the GitHub Pages hosted documentation [here](http://code.globality.com/sklearn-hierarchical-classification/).


## Installation

Requires Python 3.11+.

    pip install sklearn-hierarchical-classification

Or with [uv](https://docs.astral.sh/uv/):

    uv add sklearn-hierarchical-classification


## Usage

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

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
    base_estimator=CalibratedClassifierCV(SVC(gamma=0.001), ensemble=False),
    class_hierarchy=class_hierarchy,
)
clf.fit(X_train, y_train)  # y_train holds leaf labels, e.g. "1", "7", ...
y_pred = clf.predict(X_test)
```

The base estimator must expose `predict_proba` (or `decision_function`, when
`use_decision_function=True`). For multi-label hierarchies pass the fitted `MultiLabelBinarizer` as
`mlb`; `mlb_prediction_threshold` then takes either one threshold or one per class, and
`training_strategy="inclusive"` trains each node on out-of-subtree documents as negatives as well, which
lets tuned thresholds recover from routing mistakes higher up. `sklearn_hierarchical_classification.thresholds` tunes such
thresholds from held-out scores (per-class SCut, optionally local to each parent, or a single
label-cardinality threshold). See [examples/](./examples/) for a
complete, runnable example that also demonstrates the hierarchical evaluation metrics.

### Jupyter notebooks

Support for interactive development is built in to the `HierarchicalClassifier` class. This will enable progress bars (using the excellent [tqdm](https://pypi.python.org/pypi/tqdm) library) in various places during training and may otherwise enable more visibility into the classifier which is useful during interactive use. To enable this make sure widget extensions are enabled by running:

    jupyter nbextension enable --py --sys-prefix widgetsnbextension

You can then instantiate a classifier with the `progress_wrapper` parameter set to `tqdm_notebook`:

```python
clf = HierarchicalClassifier(
    base_estimator=svm.LinearSVC(),
    class_hierarchy=class_hierarchy,
    progress_wrapper=tqdm_notebook,
    use_decision_function=True,  # LinearSVC exposes decision_function rather than predict_proba
)
```


## Upgrading from 1.3.x

- Python 3.11+ and scikit-learn 1.6+ are required.
- `HierarchicalClassifier.fit()` no longer accepts `sample_weight`. It was accepted but never used
  (weights were silently ignored), which current scikit-learn estimator checks reject.
- A node's training set is now selected by label only. Previously a sample whose feature row was
  entirely zero was silently left out of every local classifier (and of the `n_samples` metafeature).
- On a DAG, a sample under several children of a node is now used once per child (previously once per
  *path*, which duplicated rows and, in the feature-building step, doubled feature values).
- `_select_features(X, y)` is called once per trained node with that node's rows, and must keep the
  number of columns since prediction passes unselected rows to the local classifiers.
- The `array` helpers `extract_rows_csr`, `nnz_rows_ix`, `apply_rollup_Xy` and `apply_rollup_Xy_raw`
  were internal to the old per-node matrix building and have been removed.
- In multi-label mode (`mlb`), `predict()` now returns a binary indicator matrix over `mlb.classes_` of
  the nodes visited by each sample, matching the `y` passed to `fit` and the columns of
  `predict_proba()`. It previously returned an array of root-prefixed node paths with duplicated entries.
  `predict_proba()` in that mode is now sized by `mlb.classes_`.
- Early stopping (`prediction_depth="nmlnp"` / `stopping_criteria`) is rejected together with `mlb`; it
  was silently ignored before. A callable `stopping_criteria` returns True to stop (the code always
  behaved this way; the docstring said the opposite) and is no longer consulted at the root.
- The default base estimator is `LogisticRegression(solver="lbfgs")` without the removed
  `multi_class="multinomial"` argument. On scikit-learn 1.6/1.7 this means binary local classifiers use
  one-vs-rest, so predicted probabilities (and hence `nmlnp` stopping decisions) can differ slightly
  from 1.3.x; on scikit-learn 1.8+ there is no difference.


## Development

The project is managed with [uv](https://docs.astral.sh/uv/). All tooling configuration lives in
`pyproject.toml`; linting, formatting, type-checking and tests are enforced through
[pre-commit](https://pre-commit.com/) both locally and in CI.

    uv sync --dev                       # create .venv and install the package + dev tools
    uv run pre-commit install           # run the hooks on every commit

    uv run pytest                       # full test suite (includes the slow, dataset-downloading tests)
    uv run pytest -m "not slow"         # what the pre-commit hook runs
    uv run pre-commit run --all-files   # everything CI enforces: ruff, mypy, hygiene hooks, tests


## Benchmarks

`benchmarks/bench.py` times fit/predict on synthetic sparse data and hierarchies of configurable
shape; `benchmarks/rcv1_benchmark.py` runs the classic RCV1-v2 newswire benchmark (Lewis et al.
2004: 103 topics in a 4-root hierarchy, 23,149 training and 781,265 test documents, fetched via
scikit-learn) against a flat one-vs-rest baseline on the same TF-IDF features. Results on a laptop,
`LinearSVC` base classifiers, decision-function threshold 0 (no per-category tuning):

| Model | micro-F1 | macro-F1 | hF1 | fit | predict (781k docs) |
|---|---:|---:|---:|---:|---:|
| Flat `OneVsRest(LinearSVC)` | 0.804 | 0.486 | 0.808 | 3.7 s | 5.7 s |
| `HierarchicalClassifier` (LCPN, `LinearSVC` per node) | 0.796 | 0.514 | 0.796 | 1.9 s | 2.2 s |
| `HierarchicalClassifier` + per-class thresholds tuned by 5-fold CV (`--tune`) | 0.803 | 0.586 | 0.803 | 11.1 s | 2.4 s |
| Published SVM, per-category tuned thresholds (Lewis et al. 2004) | 0.816 | 0.607 | | | |

    uv run python benchmarks/bench.py --help
    uv run python benchmarks/rcv1_benchmark.py


## Releasing

Versions are derived from git tags by [setuptools-scm](https://github.com/pypa/setuptools-scm); there
is no version string to bump in the tree. Pushing a tag builds the distributions, publishes them to
PyPI via trusted publishing, and creates a GitHub Release:

    git tag 1.4.0
    git push origin 1.4.0


## Documentation

Auto-generated documentation is provided via sphinx. To build / view:

    $ cd docs/
    $ make html
    $ open build/html/index.html


Documentation is published to GitHub pages from the `gh-pages` branch.
If you are a contributor and need to update documentation, a good starting point for getting setup is [this tutorial](https://gohugo.io/hosting-and-deployment/hosting-on-github/#deployment-of-project-pages-from-docs-folder-on-master-branch).


## Further Reading

this module is heavily influenced by the following previous work and papers:

* ["Functional Annotation of Genes Using Hierarchical Text Categorization"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf) - Kiritchenko et al. 2005
* ["Classifying web documents in a hierarchy of categories: a comprehensive study"](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.150.8859) - Ceci and Malerba 2007
* ["A survey of hierarchical classification across different application domains"](https://www.researchgate.net/publication/225716424_A_survey_of_hierarchical_classification_across_different_application_domains) - CN Silla et al. 2011
* ["A Survey of Automated Hierarchical Classification of Patents"](https://lirias.kuleuven.be/bitstream/123456789/457904/1/GomezMoens%20Mumia_book_chapter_camera_ready2014.pdf) - JC Gomez et al. 2014
* ["Evaluation Measures for Hierarchical Classification: a unified view and novel approaches"](https://arxiv.org/pdf/1306.6802.pdf) - Kosmopoulos et al. 2013
* ["Bayesian Aggregation for Hierarchical Classification"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.3312&rep=rep1&type=pdf) - Barutcuoglu et al. 2008
* ["Kaggle LSHTC4 Winning Solution"](https://kaggle2.blob.core.windows.net/forum-message-attachments/43550/1230/lshtc4.pdf) - Puurula et al. 2014
* ["Feature-Weighted Linear Stacking"](https://arxiv.org/pdf/0911.0460.pdf) - Sill et al. 2009
