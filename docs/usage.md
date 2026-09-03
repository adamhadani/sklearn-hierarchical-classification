---
title: Usage
nav_order: 2
---

# Usage
{: .no_toc }

1. TOC
{:toc}

## The class hierarchy

A hierarchy is a rooted directed acyclic graph whose nodes are class labels and whose edges
point from a class to its sub-classes. Pass it as a `networkx.DiGraph` or as an adjacency
dict mapping each parent to its children. The root is the framework-provided `ROOT` sentinel
unless you set `root=` to a node of your own graph.

```python
from networkx import DiGraph

from sklearn_hierarchical_classification.constants import ROOT

# Equivalent hierarchies
as_dict = {ROOT: ["A", "B"], "A": ["1", "7"], "B": ["C", "9"], "C": ["3", "8"]}
as_graph = DiGraph(as_dict)
```

```mermaid
graph TD
    R((ROOT)) --> A
    R --> B
    A --> 1
    A --> 7
    B --> C
    B --> 9
    C --> 3
    C --> 8
```

Intermediate nodes (`A`, `B`, `C` above) are labels like any other: a sample may be labelled
with one, and `predict` may return one when early stopping is on. Leave `class_hierarchy`
unset and `fit` builds a flat hierarchy linking every class in `y` to the root, which makes
the classifier a plain multi-class one.

{: .note }
Use node identifiers of a single type. Predictions are returned as a numpy array, and mixed
`int`/`str` labels would be coerced to strings.

`fit` checks the hierarchy before training: it must contain `root` and be acyclic (a
`ValueError` otherwise), and nodes the root cannot reach raise a warning since they can never
be predicted. Labels of `y` that are not hierarchy nodes also raise a warning: they are ignored,
which is what a typo in a label would otherwise cost silently.

### DAGs

A node may have several parents. Its training samples reach every ancestor once, and at
prediction time a node is scored once per call on the union of the samples that reached it
from all of its parents. In multi-label mode `predict_proba` reports, for a class under several
visited parents, the highest of its local scores, which is the quantity its threshold is
compared with.

## Training: one classifier per parent node

`fit` visits the hierarchy depth-first from the root and trains one local classifier at every
node that has children. The training set of a node is the samples labelled with any strict
descendant of the node; the targets are those labels rolled up to the node's children. A node
whose training targets hold a single child gets a constant predictor instead of a copy of the
base estimator, and a node with no training samples gets no classifier and a warning.

Training data is never copied per node: each local classifier is fitted on the rows of `X`
selected by index, so sparse input stays sparse and the fitted model keeps no reference to
the training set. After `fit`, `graph_` is the `DiGraph` holding a `classifier` and
`metafeatures` (`n_samples`, `n_targets`) on every trained node.

### Base estimators

`base_estimator` is any scikit-learn classifier exposing `predict_proba`, or
`decision_function` when `use_decision_function=True`. The default is
`LogisticRegression(solver="lbfgs", max_iter=1000)`.

To vary the estimator across the hierarchy, pass a dict keyed by node with the `DEFAULT`
constant as the catch-all, or a callable receiving the node and the graph as keyword arguments `node_id` and `graph`:

```python
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn_hierarchical_classification.constants import DEFAULT

clf = HierarchicalClassifier(
    base_estimator={ROOT: LinearSVC(), DEFAULT: LogisticRegression()},
    class_hierarchy=class_hierarchy,
    use_decision_function=True,
)

def estimator_for(node_id, graph):
    return LinearSVC() if graph.out_degree(node_id) > 5 else LogisticRegression()

clf = HierarchicalClassifier(base_estimator=estimator_for, class_hierarchy=class_hierarchy, use_decision_function=True)
```

The estimator is cloned for every node, so one instance can be shared. `HierarchicalClassifier`
itself clones like any scikit-learn estimator (`sklearn.base.clone`, grid search,
cross-validation), with one deliberate exception: a fitted `mlb` is passed on to the clone as is,
since it names the classes of `y` rather than holding anything learned from `X`. Estimators with
`decision_function` only (such as `LinearSVC`) need `use_decision_function=True`; a binary
`decision_function` returns one signed score, which is expanded to a score per class.

### Raw inputs and pipelines

With `feature_extraction="raw"`, `X` is a Python sequence of raw samples (strings, dicts,
anything) and the base estimator is a `Pipeline` that starts with feature extraction. Each
node's pipeline is then fitted on the node's own samples, so a text vectorizer builds a
vocabulary per node:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

clf = HierarchicalClassifier(
    base_estimator=make_pipeline(TfidfVectorizer(), LogisticRegression()),
    class_hierarchy=class_hierarchy,
    feature_extraction="raw",
)
clf.fit(documents, labels)
```

A `Pipeline` also works in the default `"preprocessed"` mode on a feature matrix, for
example to run a per-node `TruncatedSVD`; see `examples/classify_digits.py`.

## Prediction

`predict` walks each sample down from the root: at every visited node the local classifier
picks the child to descend into, and the walk ends at a leaf. Samples are batched, so each
local classifier is called once per `predict` regardless of the number of samples.

`predict_proba` returns a matrix over `classes_` (every node except the root) holding the
local score each class received along the sample's walk, and zero for classes at nodes
that were not visited. It is not a distribution over the leaves: scores at different depths
come from different classifiers.

### Scores and calibration

The scores in `predict_proba` are whatever the local classifiers report: probabilities from
their `predict_proba`, or signed margins when `use_decision_function=True` and the estimator
has a `decision_function`. They are local to each node (the score of `3` above is the
probability of `3` *given* that the walk reached `C`) and are not calibrated against each
other across nodes or depths. Two consequences:

- A float `stopping_criteria` is compared with those scores as they are: `0.7` means a
  probability with the default estimator, and a margin with an SVM under
  `use_decision_function`.
- The probability of a whole path is the product of the local probabilities along it, which
  `predict_proba` does not compute. On a tree, in single-label mode, it takes a few lines. It is
  meaningful for the predicted leaf and its siblings only: nodes the walk did not visit hold a
  zero, so leaves under another branch get zero and cannot be ranked against the predicted one.

```python
import numpy as np
from networkx import shortest_path

proba = clf.predict_proba(X_test)
column = {label: i for i, label in enumerate(clf.classes_)}

def path_probability(row, leaf):
    path = shortest_path(clf.graph_, clf.root, leaf)[1:]  # the nodes below the root
    return np.prod([proba[row, column[node]] for node in path])
```

To get probabilities from a margin-based base estimator, wrap it in scikit-learn's
`CalibratedClassifierCV`. It is cloned and fitted at every node, so keep `cv` small where nodes
have few samples; `ensemble=False` keeps a single model per node.

### Early stopping

By default prediction always ends at a leaf (`prediction_depth="mlnp"`, mandatory leaf-node
prediction). With `prediction_depth="nmlnp"` the walk may stop at an intermediate node, which
is then the prediction. `stopping_criteria` decides, and is required in this mode:

- a `float`: stop when the local score of the chosen child is below it;
- a callable `f(current_node, prediction, score)` receiving the node's attributes dict (with
  its `metafeatures`), the chosen child and its score, returning `True` to stop.

The walk never stops at the root, so every sample gets at least one label.

```python
clf = HierarchicalClassifier(
    base_estimator=LogisticRegression(),
    class_hierarchy=class_hierarchy,
    prediction_depth="nmlnp",
    stopping_criteria=0.7,
)
```

Early stopping is a single-label feature and is rejected together with `mlb`.

## Inspecting the fitted model

`graph_` is the `networkx.DiGraph` of the hierarchy, and every node that was given a
classifier carries it in its node attributes (the attribute names are in `constants`):

| Attribute | Content |
|---|---|
| `"classifier"` | The fitted local classifier: a clone of the base estimator, or (single-label mode) a constant `DummyClassifier` where the training targets held a single child. |
| `"metafeatures"` | `{"n_samples": ..., "n_targets": ...}`: the number of samples in the node's subtree (its training set, except under inclusive training, which adds the rest) and the number of distinct labels among them. |
| `"trained_classes"` | Multi-label mode only: the children that had a positive example at the node. Only these are routed to. |

```python
from sklearn_hierarchical_classification.constants import CLASSIFIER, METAFEATURES

for node, attributes in clf.graph_.nodes(data=True):
    if CLASSIFIER in attributes:
        print(node, attributes[METAFEATURES], type(attributes[CLASSIFIER]).__name__)

clf.graph_.nodes["A"][CLASSIFIER].coef_  # the weights of the classifier choosing between 1 and 7
```

Leaves carry no classifier. A parent node without one had no training sample under it: `fit`
logs a warning, and the walk of any sample routed there ends at that node. `classes_` lists
every node except the root, in the column order of `predict_proba` in single-label mode; the
multi-label columns follow `mlb.classes_`.

## Saving and loading

A fitted classifier pickles like any scikit-learn estimator; `joblib` is the usual choice:

```python
import joblib

joblib.dump(clf, "hierarchical.joblib")
clf = joblib.load("hierarchical.joblib")
```

The file holds the hierarchy with its fitted local classifiers (and `mlb`, in multi-label
mode) and nothing of the training data beyond what the base estimators themselves keep (an
`SVC` its support vectors, a nearest-neighbours model everything). The usual pickle caveats apply: load only files you
trust, with the versions of scikit-learn and of this package that wrote them.

## Progress and logging

`progress_wrapper` takes a `tqdm`-style callable (`tqdm`, `tqdm.notebook.tqdm`, ...) that is
called as `progress_wrapper(total=..., desc=...)` and wraps the training loop with it. Training
details are logged on the `HierarchicalClassifier` logger at `DEBUG` level; a node left without
training data logs a warning.

## Parameters at a glance

| Parameter | Default | Meaning |
|---|---|---|
| `base_estimator` | `None` (a `LogisticRegression` per node) | Estimator, dict by node (`DEFAULT` catch-all) or callable `(node_id=, graph=)`. |
| `class_hierarchy` | flat | `DiGraph` or adjacency dict rooted at `root`. |
| `root` | `ROOT` | Identifier of the root node. |
| `prediction_depth` | `"mlnp"` | `"nmlnp"` allows stopping at intermediate nodes. |
| `stopping_criteria` | `None` | Float or callable, required with `"nmlnp"`. |
| `algorithm` | `"lcpn"` | Local classifier per parent node. `"lcn"` is deprecated: it was never implemented and warns at `fit`. |
| `training_strategy` | `None` | `"siblings"` (default) or `"inclusive"` (multi-label only). See [Multi-label](multi-label). |
| `feature_extraction` | `"preprocessed"` | `"raw"` passes the raw samples to a pipeline base estimator. |
| `mlb` | `None` | Fitted `MultiLabelBinarizer` for multi-label targets. |
| `mlb_prediction_threshold` | `0.0` | One threshold, or one per `mlb.classes_`. |
| `mlb_min_root_predictions` | `0` | Root children forced on samples that would get none. |
| `use_decision_function` | `False` | Score with `decision_function` when the estimator has one. |
| `progress_wrapper` | `None` | `tqdm`-style wrapper for training progress. |
