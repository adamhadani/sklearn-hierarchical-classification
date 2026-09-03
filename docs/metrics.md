---
title: Metrics
nav_order: 4
---

# Hierarchical metrics

Flat precision and recall treat every mistake alike, whereas predicting a sibling of the
true class is a smaller error than predicting a class in another branch. The hierarchical
metrics of Kiritchenko et al. (2005) fix that by expanding both the true and the predicted
label sets with all of their ancestors before comparing them, so a prediction earns credit
for every ancestor it shares with the truth.

`sklearn_hierarchical_classification.metrics` provides the micro-averaged versions:

| Function | Definition |
|---|---|
| `h_precision_score(y_true, y_pred, class_hierarchy)` | shared ancestor-expanded labels / predicted ancestor-expanded labels |
| `h_recall_score(y_true, y_pred, class_hierarchy)` | shared ancestor-expanded labels / true ancestor-expanded labels |
| `h_fbeta_score(y_true, y_pred, class_hierarchy, beta=1.0)` | their F-beta |

All three take `y_true` and `y_pred` as binary indicator matrices whose columns are the nodes
of `class_hierarchy` (a `DiGraph` whose nodes are the column indices), and a `root` that is
excluded from the counts. `fill_ancestors(y, graph, root)` is the expansion step on its own.

## Single-label predictions

For single-label targets, the `multi_labeled` context manager builds the indicator matrices
and the relabelled graph from label arrays and the fitted classifier's `graph_`:

```python
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled

with multi_labeled(y_test, y_pred, clf.graph_) as (y_test_, y_pred_, graph_):
    print(h_fbeta_score(y_test_, y_pred_, graph_))
```

## Multi-label predictions

In multi-label mode `predict` already returns an indicator matrix over `mlb.classes_`, so
relabel the hierarchy to column indices once and call the metrics directly:

```python
from networkx import relabel_nodes

graph_by_column = relabel_nodes(graph, {label: column for column, label in enumerate(mlb.classes_)})
print(h_precision_score(Y_test, Y_pred, graph_by_column))
```

The graph may be a DAG: a node with several parents contributes all of its ancestors.
