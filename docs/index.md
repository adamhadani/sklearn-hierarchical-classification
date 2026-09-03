---
title: Home
layout: home
nav_order: 1
---

# sklearn-hierarchical-classification

Hierarchical classification for [scikit-learn](https://scikit-learn.org/): a meta-estimator that
trains a local classifier at every parent node of a class hierarchy and predicts by walking the
hierarchy top-down.
{: .fs-6 .fw-300 }

[Get started](usage){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/adamhadani/sklearn-hierarchical-classification){: .btn .fs-5 .mb-4 .mb-md-0 }

---

Many classification targets have structure: product categories, topic taxonomies, gene
functions, book genres. `HierarchicalClassifier` takes that structure as a tree or DAG
(a `networkx.DiGraph` or a plain adjacency dict) and turns any scikit-learn classifier into
a hierarchical one, following the "local classifier per parent node" scheme of Silla and
Freitas (2011).

- **scikit-learn compatible.** `fit` / `predict` / `predict_proba`, `get_params` /
  `set_params`, pickling, sparse input, and scikit-learn's own estimator checks.
- **Any base classifier**, one per node if you like, or a full `Pipeline` over raw inputs
  such as text.
- **Multi-label hierarchies** through a `MultiLabelBinarizer`, with per-class decision
  thresholds, a training strategy that lets nodes reject mis-routed samples, and a
  `thresholds` module that tunes thresholds from held-out scores.
- **Hierarchical metrics** (hP, hR, hF-beta) in `metrics`.
- **Benchmarked** on RCV1-v2 and GermEval 2019, with numbers that match or beat the published
  SVM baselines: see [Benchmarks](benchmarks).

## Installation

Python 3.11 or newer.

```bash
pip install sklearn-hierarchical-classification
# or
uv add sklearn-hierarchical-classification
```

## A first example

```python
from sklearn.linear_model import LogisticRegression

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT

class_hierarchy = {
    ROOT: ["A", "B"],
    "A": ["1", "7"],
    "B": ["3", "8", "9"],
}

clf = HierarchicalClassifier(
    base_estimator=LogisticRegression(),
    class_hierarchy=class_hierarchy,
)
clf.fit(X_train, y_train)   # y_train holds leaf labels: "1", "7", "3", ...
y_pred = clf.predict(X_test)
```

Three local classifiers are trained: one at the root choosing between `A` and `B`, one at `A`
choosing between `1` and `7`, and one at `B`. Prediction walks down from the root and returns
the leaf it ends at.

## Where next

- [Usage](usage): hierarchies, base estimators, raw inputs, early stopping.
- [Multi-label](multi-label): indicator targets, training strategies, thresholds and tuning.
- [Metrics](metrics): hierarchical precision, recall and F-beta.
- [API reference](api): every public class and function.
- [Upgrading from 1.3.x](upgrading) if you used the original Globality release.

## Credits

The package was created by Globality Engineering and released under the Apache License 2.0.
This is its continued development; the original repository is archived.
