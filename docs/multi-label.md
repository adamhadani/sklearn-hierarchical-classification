---
title: Multi-label
nav_order: 3
---

# Multi-label classification
{: .no_toc }

1. TOC
{:toc}

A sample may carry several labels, at any depth of the hierarchy. Fit a
`MultiLabelBinarizer` on the label sets, pass it as `mlb`, and give `fit` the binary
indicator matrix it produces (dense or sparse). `predict` then returns an indicator matrix over
`mlb.classes_` of every node each sample was routed to, in the same format.

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

mlb = MultiLabelBinarizer().fit(label_sets)          # e.g. [["fruit", "apple"], ["veg"], ...]
Y = mlb.transform(label_sets)

clf = HierarchicalClassifier(
    base_estimator=OneVsRestClassifier(LinearSVC()),
    class_hierarchy=class_hierarchy,
    mlb=mlb,
    use_decision_function=True,
    training_strategy="inclusive",
    mlb_min_root_predictions=1,
)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)                        # indicator matrix over mlb.classes_
```

The local classifier at each node is now a multi-label one: it receives indicator targets
over the node's children and must score every child independently, which is what
`OneVsRestClassifier` does. Labels on internal nodes are fine; a label the binarizer does not
know is ignored.

## The walk

Prediction starts at the root and, at every visited node, descends into every child whose
score exceeds its threshold. Nodes are visited in topological order and each is scored once on
all the samples that reached it, from any parent. The prediction is the set of visited nodes.

A node only routes to children that had a positive example in its training set (kept in
`graph_.nodes[node]["trained_classes"]`): a one-vs-rest estimator gives an unlearned class a
constant score of zero, which any negative threshold would otherwise select for every sample.

### Thresholds

`mlb_prediction_threshold` is compared with the local scores: with `predict_proba` scores a
value like `0.5`, with `decision_function` scores a value around `0`. Pass a single float or
an array with one threshold per `mlb.classes_` column, for example thresholds tuned per class
on held-out data (see below). Thresholds are read at prediction time, so they can be set on a
fitted model with `set_params` without refitting.

Predicting with `mlb_prediction_threshold=-np.inf` visits every node learned at fit and makes
`predict_proba` return the score of every class for every sample. That matrix is the input to
threshold tuning.

### Root fallback

A sample whose scores clear no threshold at the root gets no label at all.
`mlb_min_root_predictions=k` forces the `k` best-scoring root children on such samples, and
the walk continues below them. Ties are broken in column order.

## Training strategies

The training set of a node is, by default, the samples of its subtree (`"siblings"`): the
classifier learns to tell the children apart, and its scores mean nothing for samples from
outside the subtree, which it never saw.

With `training_strategy="inclusive"` every sample outside the subtree also joins the node's
training set, as an all-negative row. The local classifier then learns to reject samples a
parent mis-routes to it, at the cost of training every node on the whole dataset. On the
benchmarks this is worth about two points of micro-F1; for a tree it makes the local
classifiers equivalent to a flat one-vs-rest model whose predictions are made consistent
top-down. Inclusive training requires `mlb`.

## Tuning thresholds

`sklearn_hierarchical_classification.thresholds` turns a matrix of held-out all-node scores
(out-of-fold, or from a development split) into thresholds. A typical loop:

```python
import numpy as np
from sklearn.model_selection import KFold

from sklearn_hierarchical_classification.thresholds import routed_thresholds

scores, scored = np.zeros(Y.shape), np.zeros(Y.shape, dtype=bool)
for fit_rows, score_rows in KFold(5, shuffle=True, random_state=0).split(X):
    fold = clone(clf).set_params(mlb_prediction_threshold=-np.inf).fit(X[fit_rows], Y[fit_rows])
    scores[score_rows] = fold.predict_proba(X[score_rows])
    scored[score_rows] = Y[fit_rows].any(axis=0)   # classes without positives were not learned

thresholds = routed_thresholds(scores, Y, graph=clf.graph_, classes=mlb.classes_, scored=scored, min_root=1)
clf.set_params(mlb_prediction_threshold=thresholds)
```

Three policies are provided:

- **`scut_thresholds`**: per-class thresholds maximising F1 (SCut, Yang 1999). With `graph`
  and `classes`, each class is tuned only on the samples truly under its parent, the population
  its local classifier was trained on. Tune *locally* like this: a global SCut over all samples
  is worse than no tuning at all on RCV1.
- **`routed_thresholds`**: per-class thresholds tuned sequentially top-down, each class on the
  samples its already-tuned parent actually *predicts*, including the root fallback through
  `min_root`. This models the walk itself and beats local SCut on micro-F1 at equal macro-F1
  on RCV1. It needs enough held-out positives per class; on GermEval, where most of the 343
  labels have a handful, every per-class scheme loses to one scalar threshold chosen on a grid.
- **`label_cardinality_threshold`**: one scalar whose predicted number of labels per sample
  matches a target cardinality, typically the training set's.

`route(scores, thresholds, graph, classes, min_root=..., scored=...)` emulates the classifier's
walk on a score matrix and returns the indicator matrix `predict` would produce, so candidate
policies can be compared on held-out scores without refitting. It is exact on trees and a
superset on DAGs. The `scored` mask marks cells that carry a real score; cells of classes not
learned in a fold are placeholders and are neither tuned on nor predicted.

{: .warning }
Do not tune thresholds on the data you report results on. Use out-of-fold scores of the
training set or a separate development split, and score the test set once.
