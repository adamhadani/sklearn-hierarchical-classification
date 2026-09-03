---
title: API reference
nav_order: 6
---

# API reference
{: .no_toc }

1. TOC
{:toc}

## `sklearn_hierarchical_classification.classifier`

### `HierarchicalClassifier`

```python
HierarchicalClassifier(
    base_estimator=None,
    class_hierarchy=None,
    prediction_depth="mlnp",
    algorithm="lcpn",
    training_strategy=None,
    stopping_criteria=None,
    root=ROOT,
    progress_wrapper=None,
    feature_extraction="preprocessed",
    mlb=None,
    mlb_prediction_threshold=0.0,
    mlb_min_root_predictions=0,
    use_decision_function=False,
)
```

A `MetaEstimatorMixin, ClassifierMixin, BaseEstimator`. Parameters are described in
[Usage](usage#parameters-at-a-glance) and [Multi-label](multi-label); they are validated at
`fit`, which raises `TypeError` for inconsistent combinations (for example `"inclusive"`
without `mlb`, or early stopping with `mlb`). `algorithm="lcn"` and the training strategies
reserved for it are deprecated and raise a `FutureWarning` at `fit`; see [Upgrading](upgrading).

**Methods**

| | |
|---|---|
| `fit(X, y)` | `X` is a dense or sparse matrix, or a sequence of raw samples in `"raw"` mode. `y` holds one label per sample, or the indicator matrix of `mlb` (dense or sparse). Returns `self`. |
| `predict(X)` | The deepest node reached by each sample's walk; in multi-label mode an integer indicator matrix over `mlb.classes_` of the nodes visited. |
| `predict_proba(X)` | Local scores of every class along each sample's walk, columns following `classes_` (or `mlb.classes_`); zero for classes at nodes not visited. |

**Fitted attributes**

| | |
|---|---|
| `classes_` | List of every hierarchy node except the root. |
| `n_classes_` | `len(classes_)`. |
| `n_features_in_` | Number of input features (not set in `"raw"` mode). |
| `class_hierarchy_` | The hierarchy as given, or the flat one built from `y`. |
| `graph_` | `networkx.DiGraph` of the hierarchy. Every trained node holds `"classifier"`, `"metafeatures"` (`n_samples`, `n_targets`) and, in multi-label mode, `"trained_classes"`. |

## `sklearn_hierarchical_classification.thresholds`

All functions take `scores` and `y` of shape `(n_samples, n_classes)` with aligned columns,
typically the output of `predict_proba` with `mlb_prediction_threshold=-np.inf` on held-out
samples and the matching indicator targets. An optional boolean `scored` of the same shape
marks the cells that carry a real score. See [Tuning thresholds](multi-label#tuning-thresholds).

| | |
|---|---|
| `scut_thresholds(scores, y, graph=None, classes=None, root=ROOT, scored=None)` | Per-class F1-maximising thresholds. With `graph` and `classes` (the node of each column), each class is tuned on the samples truly under its parent. `inf` for classes without positives. |
| `routed_thresholds(scores, y, graph, classes, root=ROOT, scored=None, min_root=0)` | Per-class thresholds tuned top-down on the samples routed to each class by its already-tuned parent, with the root fallback of `min_root`. |
| `route(scores, thresholds, graph, classes, root=ROOT, min_root=0, scored=None)` | The indicator matrix `predict` would return for those scores and thresholds. Exact on trees, a superset on DAGs. |
| `label_cardinality_threshold(scores, target_cardinality, candidates=None)` | One threshold whose predicted labels per sample come closest to `target_cardinality`. |
| `best_f1_threshold(scores, y)` | The F1-maximising threshold on one column; `inf` without positives. |

## `sklearn_hierarchical_classification.metrics`

| | |
|---|---|
| `h_precision_score(y_true, y_pred, class_hierarchy, root=ROOT)` | Micro-averaged hierarchical precision. |
| `h_recall_score(y_true, y_pred, class_hierarchy, root=ROOT)` | Micro-averaged hierarchical recall. |
| `h_fbeta_score(y_true, y_pred, class_hierarchy, beta=1.0, root=ROOT)` | Micro-averaged hierarchical F-beta. |
| `fill_ancestors(y, graph, root, copy=True)` | Set the ancestors of every marked label in an indicator matrix. |
| `multi_labeled(y_true, y_pred, graph)` | Context manager yielding indicator matrices and a relabelled graph for single-label arrays. |

`y_true` and `y_pred` are indicator matrices whose columns are the nodes of `class_hierarchy`,
a `DiGraph` over column indices (with `root` left as is).

## `sklearn_hierarchical_classification.graph`

Helpers over `networkx.DiGraph` hierarchies.

| | |
|---|---|
| `make_flat_hierarchy(targets, root)` | Adjacency dict linking every target to `root`. |
| `root_nodes(graph)` | Nodes with no parents. |
| `terminal_nodes(graph)` | Nodes with no children. |
| `children_by_descendant(graph, source)` | Map from every strict descendant of `source` to the children of `source` it lies under. |
| `rollup_nodes(graph, source, targets, mlb=None)` | For each target, the children of `source` it rolls up to (`[]` outside the subtree). |
| `rollup_targets(child_of, targets, mlb=None)` | The same from a precomputed `children_by_descendant` map. |

## `sklearn_hierarchical_classification.array`

| | |
|---|---|
| `flatten_list(lst)` | Flatten one level of nesting. |
| `nnz_columns_count(X)` | Number of columns with any non-zero. |
| `top_k_mask(scores, k)` | Boolean mask of the `k` highest cells per row, ties broken in column order. |

## `sklearn_hierarchical_classification.constants`

| | |
|---|---|
| `ROOT` | `"<ROOT>"`, the default artificial root node. |
| `DEFAULT` | `"default"`, the catch-all key of a `base_estimator` dict. |
| `CLASSIFIER`, `METAFEATURES`, `TRAINED_CLASSES` | The `graph_` node attribute names. |
