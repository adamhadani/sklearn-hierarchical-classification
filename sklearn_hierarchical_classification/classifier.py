"""
Hierarchical classifier interface.

"""

import warnings

import numpy as np
from networkx import DiGraph, descendants, dfs_preorder_nodes, is_directed_acyclic_graph, topological_sort
from scipy.sparse import issparse
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    clone,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from sklearn_hierarchical_classification.array import flatten_list, nnz_columns_count, top_k_mask
from sklearn_hierarchical_classification.constants import (
    CLASSIFIER,
    DEFAULT,
    METAFEATURES,
    ROOT,
    TRAINED_CLASSES,
)
from sklearn_hierarchical_classification.decorators import logger
from sklearn_hierarchical_classification.dummy import DummyProgress
from sklearn_hierarchical_classification.graph import children_by_descendant, make_flat_hierarchy, rollup_targets
from sklearn_hierarchical_classification.validation import is_estimator, validate_parameters


class _PredictionState:
    """Per-sample bookkeeping of one top-down prediction pass."""

    def __init__(self, n_samples, root, n_columns, column_index):
        self.last = np.full(n_samples, root, dtype=object)  # deepest node reached so far, per sample
        self.visits = []  # (node, rows) in visiting order; the root is never recorded
        self.class_proba = None if n_columns is None else np.zeros((n_samples, n_columns), dtype=np.float64)
        self.scored = None if n_columns is None else np.zeros((n_samples, n_columns), dtype=bool)
        self.column_index = column_index
        self.thresholds = None  # per-column prediction thresholds, multi-label mode only

    def columns_of(self, classes):
        """Score-matrix columns of a local classifier's classes."""
        try:
            return np.fromiter((self.column_index[class_] for class_ in classes), dtype=np.intp, count=len(classes))
        except KeyError as error:
            raise ValueError(
                f"Local classifier predicts class {error.args[0]!r}, which is not a column of the class hierarchy"
            ) from None

    def record(self, node, rows):
        """Mark `rows` as having reached `node`; returns `rows` for chaining."""
        self.last[rows] = node
        self.visits.append((node, rows))
        return rows

    def set_scores(self, rows, columns, scores):
        if self.class_proba is not None:
            self.class_proba[rows[:, None], columns] = scores

    def max_scores(self, rows, columns, scores):
        """Record scores, keeping the highest for a cell scored by several parents (unscored cells stay 0)."""
        if self.class_proba is None:
            return
        cells = (rows[:, None], columns)
        self.class_proba[cells] = np.where(self.scored[cells], np.maximum(self.class_proba[cells], scores), scores)
        self.scored[cells] = True


def _rows_by_label(y, columns=None):
    """
    Map each label to the sorted row indices carrying it.

    `y` is either 1-D (one label per row) or, when `columns` names its columns, a 2-D binary
    indicator matrix (one group per column, keyed by that column's label).

    """
    if columns is not None:
        return {label: np.flatnonzero(y[:, column]) for column, label in enumerate(columns)}
    if len(y) == 0:
        return {}
    order = np.argsort(y, kind="stable")
    labels, starts = np.unique(y[order], return_index=True)
    return dict(zip(labels, np.split(order, starts[1:]), strict=True))


@logger
class HierarchicalClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """Hierarchical classification strategy

    Hierarchical classification deals with the scenario where our target classes have
    inherent structure that can be represented as a tree or a directed acyclic graph (DAG),
    with nodes representing the target classes themselves, and edges representing their inter-relatedness,
    e.g "IS A" semantics.

    Within this general framework, several distinctions can be made based on a few key modelling decisions:

    - Multi-label classification - Do we support classifying into more than a single target class/label
    - Mandatory / Non-mandatory leaf node prediction - Do we require that classification always results with
        classes corresponding to leaf nodes, or can intermediate nodes also be treated as valid output predictions.
    - Local classifiers - the local (or "base") classifiers can theoretically be chosen to be of any kind, but we
        distinguish between three main modes of local classification:
            * "One classifier per parent node" - where each non-leaf node can be fitted with a multi-class
                classifier to predict which one of its child nodes is relevant for given example.
            * "One classifier per node" - where each node is fitted with a binary "membership" classifier which
                returns a binary (or a probability) score indicating the fitness for that node and the current
                example.
            * Global / "big bang" classifiers - where a single classifier predicts the full path in the hierarchy
                for a given example.

    The nomenclature used here is based on the framework outlined in [1].

    Parameters
    ----------
    base_estimator : classifier object, function, dict, or None
        A scikit-learn compatible classifier object implementing "fit" and "predict_proba" to be used as the
        base classifier.
        If a callable function is given, it will be called to evaluate which classifier to instantiate for
        current node. The function will be called with the current node and the graph instance.
        Alternatively, a dictionary mapping classes to classifier objects can be given. In this case,
        when building the classifier tree, the dictionary will be consulted and if a key is found matching
        a particular node, the base classifier pointed to in the dict will be used. Since this is most often
        useful for specifying classifiers on only a handlful of objects, a special "DEFAULT" key can be used to
        set the base classifier to use as a "catch all".
        If not provided, a base estimator will be chosen by the framework using various meta-learning
        heuristics (WIP).

    class_hierarchy : networkx.DiGraph object, or dict-of-dicts adjacency representation (see examples)
        A directed graph which represents the target classes and their relations. Must be a tree/DAG (no cycles).
        If not provided, this will be initialized during the `fit` operation into a trivial graph structure linking
        all classes given in `y` to an artificial "ROOT" node.

    prediction_depth : "mlnp", "nmlnp"
        Prediction depth requirements. This corresponds to whether we wish the classifier to always terminate at
        a leaf node (mandatory leaf-node prediction, "mlnp"), or wish to support early termination via some
        stopping criteria (non-mandatory leaf-node prediction, "nmlnp"). When "nmlnp" is specified, the
        stopping_criteria parameter is used to control the behaviour of the classifier.

    algorithm : "lcn", "lcpn"
        The algorithm type to use for building the hierarchical classification, according to the
        taxonomy defined in [1].

        "lcpn" (which is the default) stands for "local classifier per parent node". Under this model,
        a multi-class classifier is trained at each parent node, to distinguish between each child nodes.

        "lcn", which stands for "local classifier per node". Under this model, a binary classifier is trained
        at each node. Under this model, a further distinction is made based on how the training data set is constructed.
        This is controlled by the "training_strategy" parameter.

    training_strategy: "exclusive", "less_exclusive", "inclusive", "less_inclusive",
                       "siblings", "exclusive_siblings", or None.
        Dictates how the training set of each local classifier is constructed (terminology per [1]).
        With "lcpn" (the default algorithm) the choice is between "siblings" (the default when None: a node's
        classifier is trained on the documents of its subtree only, so it learns to tell the children apart)
        and "inclusive" (documents from outside the subtree are added as negatives for every child, so the
        classifier's scores are also calibrated for documents that a parent may route to it by mistake;
        requires `mlb`). The full set of values is reserved for the "lcn" algorithm.

    stopping_criteria: function, float, or None.
        This parameter is used when the "prediction_depth" parameter is set to "nmlnp", and is used to evaluate
        at a given node whether classification should terminate or continue further down the hierarchy.

        When set to a float, the prediction will stop if the reported confidence at current classifier is below
        the provided value.

        When set to a function, the callback function will be called with the current node attributes,
        including its metafeatures, and the current classification results.
        This allows the user to define arbitrary logic that can decide whether classification should stop at
        the current node or continue. The function should return True if classification should stop at the
        current node, or False if it should continue to the predicted child. It is never consulted at the
        root node. Early stopping is not available in multi-label mode (`mlb`).

    root : integer, string
        The unique identifier for the qualified root node in the class hierarchy. The hierarchical classifier
        assumes that the given class hierarchy graph is a rooted DAG, e.g has a single designated root node
        of in-degree 0. This node is associated with a special identifier which defaults to a framework provided one,
        but can be overridden by user in some cases, e.g if the original taxonomy is already rooted and there"s no need
        for injecting an artificial root node.

    progress_wrapper : callable or None
        A `tqdm`-style callable, invoked as `progress_wrapper(total=..., desc=...)`, wrapping the training loop to
        display progress updates, which is useful in interactive environments such as a Jupyter notebook. Common
        values are `tqdm.tqdm` and `tqdm.notebook.tqdm`.

    feature_extraction : "preprocessed", "raw"
        Determines the feature extraction policy the classifier uses.
        When set to "raw", the classifier will expect the raw training examples are passed in to `.fit()` and `.train()`
        as X. This means that the base_estimator should point to a sklearn Pipeline that includes feature extraction.
        When set to "preprocessed", the classifier will expect X to be a pre-computed feature (sparse) matrix.

    mlb : MultiLabelBinarizer or None
        For multi-label classification, the MultiLabelBinarizer instance that was used for creating the y variable.

    mlb_prediction_threshold : float, or array-like of shape [n_classes]
        For multi-label prediction tasks (when `mlb` is set to a MultiLabelBinarizer instance), the score above which
        a child node is considered predicted and descended into. Either a single threshold or one per class, in the
        order of `mlb.classes_` (e.g. per-class thresholds tuned on held-out data). Defaults to zero. To obtain the
        scores of every node for such tuning, predict with `mlb_prediction_threshold=-np.inf`, which visits every
        node learned at fit.

    mlb_min_root_predictions : int
        For multi-label prediction tasks, the minimum number of children of the root predicted for every sample:
        when fewer clear their thresholds, the best-scoring children are taken anyway (and descended into), so
        that no sample is left without a top-level label. Defaults to zero (no such guarantee).

    use_decision_function : bool
        Some classifiers (e.g. sklearn.svm.SVC) expose a `.decision_function()` method which would take in the
        feature matrix X and return a set of per-sample scores, corresponding to each label. Setting this to True
        would attempt to use this method when it is exposed by the base classifier.

    Attributes
    ----------
    classes_ : array, shape = [`n_classes`]
        Flat array of class labels

    References
    ----------

    .. [1] CN Silla et al., "A survey of hierarchical classification across
           different application domains", 2011.

    """

    def __init__(
        self,
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
    ):
        self.base_estimator = base_estimator
        self.class_hierarchy = class_hierarchy
        self.prediction_depth = prediction_depth
        self.algorithm = algorithm
        self.training_strategy = training_strategy
        self.stopping_criteria = stopping_criteria
        self.root = root
        self.progress_wrapper = progress_wrapper
        self.feature_extraction = feature_extraction
        self.mlb = mlb
        self.mlb_prediction_threshold = mlb_prediction_threshold
        self.mlb_min_root_predictions = mlb_min_root_predictions
        self.use_decision_function = use_decision_function

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def fit(self, X, y=None):
        """Fit underlying classifiers.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets, or, when `mlb` is set, the binary indicator matrix produced by that
            `MultiLabelBinarizer` (dense or sparse) for multi-label classification.

        Returns
        -------
        self

        """
        if self.mlb is not None and issparse(y):
            # A binary indicator matrix (e.g. from MultiLabelBinarizer(sparse_output=True) or fetch_rcv1)
            y = y.toarray()

        if self.feature_extraction == "raw":
            # In raw mode, only validate targets (y) format and
            # that targets and training data (X) are of same cardinality, since
            # X will in general not be a 2D feature matrix, but rather the raw training examples,
            # e.g. text snippets or images.
            y = check_array(y, ensure_all_finite=True, ensure_2d=False, dtype=None)
            if len(X) != y.shape[0]:
                raise ValueError("bad input shape: len(X) != y.shape[0]")
        else:
            X, y = validate_data(self, X, y, accept_sparse="csr", multi_output=self.mlb is not None)

        check_classification_targets(y)

        # Check that parameter assignment is consistent
        self._check_parameters()
        if self.mlb is not None:
            self._class_thresholds()

        # Initialize NetworkX Graph from input class hierarchy
        self.class_hierarchy_ = self.class_hierarchy or make_flat_hierarchy(list(np.unique(y)), root=self.root)
        self.graph_ = DiGraph(self.class_hierarchy_)
        self._check_hierarchy()
        self._check_labels(y)
        self.classes_ = [node for node in self.graph_.nodes() if node != self.root]

        with self._progress(total=self.graph_.number_of_nodes(), desc="Training local classifiers") as progress:
            self._train_local_classifiers(X, y, progress=progress)

        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : array-like, shape = [n_samples, ] or [n_samples, n_classes]
            Predicted targets: the deepest node reached by each sample's top-down walk. In
            multi-label mode (`mlb` is set), a binary indicator matrix over `mlb.classes_` of the
            nodes visited (the root excluded), in the same format as the `y` passed to `fit`.

        Nb. with `prediction_depth="nmlnp"` predictions mix intermediate and leaf nodes, so use
        node identifiers of a single type; mixed int/str labels are coerced to strings by numpy.

        """
        check_is_fitted(self, "graph_")
        X = self._check_predict_input(X)
        state = self._predict_top_down(X, n_columns=None)
        if self.mlb is not None:
            return self._visits_as_indicator(state, n_samples=self._n_samples(X))
        return np.asarray(state.last.tolist())

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            The local classifier scores seen for each class along the top-down walk of each
            sample, zero for classes at nodes that were not visited. Columns follow `classes_`,
            or `mlb.classes_` in multi-label mode, where each visited node contributes the scores of
            its children and a class under several visited parents reports the highest: the
            quantity its threshold is compared with, since it is reached if any parent passes it.
        """
        check_is_fitted(self, "graph_")
        X = self._check_predict_input(X)
        n_columns = len(self.mlb.classes_) if self.mlb is not None else self.n_classes_
        return self._predict_top_down(X, n_columns=n_columns).class_proba

    def __sklearn_clone__(self):
        """Clone like `sklearn.base.clone`, except that a fitted `mlb` is passed on as is: it names the
        classes of `y` and is never fitted here, so resetting it would leave the clone unable to fit."""
        cloned = super().__sklearn_clone__()
        cloned.mlb = self.mlb
        return cloned

    def _check_hierarchy(self):
        """The hierarchy must be a DAG containing `root`; nodes the root cannot reach are never predicted."""
        if self.root not in self.graph_:
            raise ValueError(f"'root' {self.root!r} is not a node of the class hierarchy")
        if not is_directed_acyclic_graph(self.graph_):
            raise ValueError("The class hierarchy contains a cycle; it must be a tree or a DAG")
        unreachable = sorted(set(self.graph_.nodes) - descendants(self.graph_, self.root) - {self.root}, key=str)
        if unreachable:
            warnings.warn(
                f"Nodes {unreachable} of the class hierarchy cannot be reached from the root {self.root!r} and "
                "will never be predicted",
                UserWarning,
                stacklevel=3,
            )

    def _check_labels(self, y):
        """Labels of `y` (in multi-label mode, classes with a positive) that are not hierarchy nodes are
        ignored: warn, since a typo in a label would otherwise silently cost its samples."""
        if self.class_hierarchy is None:
            return  # the flat hierarchy was just built from these labels
        if self.mlb is None:
            labels = np.unique(y).tolist()
        else:
            labels = np.asarray(self.mlb.classes_)[np.asarray(y).any(axis=0)].tolist()
        unknown = [label for label in labels if label not in self.graph_]
        if unknown:
            warnings.warn(
                f"Labels {unknown} of `y` are not nodes of the class hierarchy and are ignored",
                UserWarning,
                stacklevel=3,
            )

    def _check_predict_input(self, X):
        if self.feature_extraction == "raw":
            return X
        return validate_data(self, X, accept_sparse="csr", reset=False)

    def _predict_top_down(self, X, n_columns):
        """Walk the hierarchy top-down for every sample at once.

        Nodes are processed in topological order, each scoring in a single call exactly the
        samples that reached it (from any number of parents), then handing each sample on to the
        child (or children, in multi-label mode) it selects. `n_columns` sizes the score matrix,
        or is None when only the reached nodes are needed.

        """
        n_samples = self._n_samples(X)
        state = _PredictionState(n_samples, root=self.root, n_columns=n_columns, column_index=self._column_index())
        state.thresholds = self._class_thresholds() if self.mlb is not None else None
        if n_samples == 0:
            return state
        if CLASSIFIER not in self.graph_.nodes[self.root]:
            raise ValueError(
                "No local classifier at the root node: fit found no training samples labeled with a "
                "descendant of the root, so nothing can be predicted"
            )

        descend = self._descend_multi_label if self.mlb is not None else self._descend_single_label
        inbox = {self.root: [np.arange(n_samples)]}
        for node_id in topological_sort(self.graph_):
            arrived = inbox.pop(node_id, None)
            clf = self.graph_.nodes[node_id].get(CLASSIFIER)
            if arrived is None or clf is None:
                # Nothing reached this node, or the walk of what did ends here
                continue
            rows = np.unique(np.concatenate(arrived))
            X_rows = X if len(rows) == n_samples else self._rows(X, rows)
            for child, child_rows in descend(node_id, clf, rows, self._local_scores(clf, X_rows), state):
                inbox.setdefault(child, []).append(child_rows)
        return state

    def _column_index(self):
        """Score-matrix column of each class: positions of `classes_`, or of the `mlb` columns."""
        if self.mlb is not None:
            return {column: column for column in range(len(self.mlb.classes_))}
        return {class_: column for column, class_ in enumerate(self.classes_)}

    def _descend_single_label(self, node_id, clf, rows, scores, state):
        """Record local scores, pick the best child per sample, and return the (child, rows) to visit next."""
        state.set_scores(rows, state.columns_of(clf.classes_), scores)
        best = scores.argmax(axis=1)
        predictions = np.asarray(clf.classes_)[best]
        kept = np.flatnonzero(~self._should_stop(node_id, predictions, scores[np.arange(len(rows)), best]))
        return [
            (clf.classes_[column], state.record(clf.classes_[column], rows[kept[positions]]))
            for column, positions in _rows_by_label(best[kept]).items()
        ]

    def _descend_multi_label(self, node_id, clf, rows, scores, state):
        """Accumulate local scores and descend into every child of this node scoring above its threshold."""
        columns = state.columns_of(clf.classes_)

        # Only local classes that are children of this node, and were learned here, are recorded and route
        # samples: a one-vs-rest classifier also carries (constant) predictors for every other column
        children = set(self.graph_.successors(node_id))
        children &= self.graph_.nodes[node_id].get(TRAINED_CLASSES, children)
        local = [local_column for local_column, column in enumerate(columns) if self.mlb.classes_[column] in children]
        child_scores = scores[:, local]
        state.max_scores(rows, columns[local], child_scores)
        selected = child_scores > state.thresholds[columns[local]]
        if node_id == self.root and self.mlb_min_root_predictions:
            # Samples with too few root labels get their best-scoring children regardless of thresholds
            short = selected.sum(axis=1) < self.mlb_min_root_predictions
            selected[short] |= top_k_mask(child_scores[short], self.mlb_min_root_predictions)

        next_nodes = []
        for k, local_column in enumerate(local):
            child = self.mlb.classes_[columns[local_column]]
            if selected[:, k].any():
                next_nodes.append((child, state.record(child, rows[selected[:, k]])))
        return next_nodes

    def _class_thresholds(self):
        """One prediction threshold per `mlb.classes_` column, validated (thresholds may be set after fit)."""
        try:
            thresholds = np.asarray(self.mlb_prediction_threshold, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("'mlb_prediction_threshold' must be a float or a 1-D array-like of floats") from None
        n_classes = len(self.mlb.classes_)
        if thresholds.ndim > 1 or np.isnan(thresholds).any() or (thresholds.ndim == 1 and len(thresholds) != n_classes):
            raise ValueError(
                f"'mlb_prediction_threshold' must be a float or one threshold per class ({n_classes}, in the order "
                f"of mlb.classes_); got shape {thresholds.shape}"
            )
        return np.broadcast_to(thresholds, (n_classes,))

    def _visits_as_indicator(self, state, n_samples):
        indicator = np.zeros((n_samples, len(self.mlb.classes_)), dtype=int)
        column_of = {label: column for column, label in enumerate(self.mlb.classes_)}
        for node, rows in state.visits:
            indicator[rows, column_of[node]] = 1
        return indicator

    def _should_stop(self, node_id, predictions, scores):
        """
        Boolean mask of the samples whose top-down walk terminates at `node_id`, per the
        "prediction_depth" and "stopping_criteria" parameters. The walk never stops at the
        (artificial) root, so every sample gets at least one prediction.

        """
        if self.prediction_depth != "nmlnp" or node_id == self.root:
            return np.zeros(len(predictions), dtype=bool)

        if callable(self.stopping_criteria):
            node = self.graph_.nodes[node_id]
            return np.array(
                [
                    bool(self.stopping_criteria(current_node=node, prediction=prediction, score=score))
                    for prediction, score in zip(predictions, scores, strict=True)
                ],
                dtype=bool,
            )

        if isinstance(self.stopping_criteria, float):
            return scores < self.stopping_criteria
        return np.zeros(len(predictions), dtype=bool)

    def _local_scores(self, clf, X):
        """
        Score a batch of samples with the local classifier at a node.

        Returns a 2-D array of shape [n_samples, n_local_classes] aligned with ``clf.classes_``,
        whether the scores come from ``decision_function`` or ``predict_proba``.

        """
        if self.use_decision_function and hasattr(clf, "decision_function"):
            scores = np.asarray(clf.decision_function(X))
            if scores.ndim == 1:
                # A binary decision_function returns one signed score per sample, for classes_[1]
                scores = np.column_stack([-scores, scores]) if len(clf.classes_) == 2 else scores[:, None]
        else:
            scores = np.asarray(clf.predict_proba(X))

        expected = (self._n_samples(X), len(clf.classes_))
        if scores.shape != expected:
            raise ValueError(
                f"Local classifier {type(clf).__name__} returned scores of shape {scores.shape}, expected {expected}"
            )
        return scores

    @property
    def n_classes_(self):
        return len(self.classes_)

    def _check_parameters(self):
        """Check the parameter assignment is valid and internally consistent."""
        validate_parameters(self)

    def _n_samples(self, X):
        return len(X) if self.feature_extraction == "raw" else X.shape[0]

    def _rows(self, X, idx):
        """Select rows of `X` by index, whether `X` is a feature matrix or a raw sample sequence."""
        if self.feature_extraction == "raw":
            return [X[i] for i in idx]
        return X[idx]

    def _train_local_classifiers(self, X, y, progress):
        """Train the local classifier at every node, in depth-first order from the root.

        Each node's training set is the set of samples labeled with a strict descendant of that node
        (samples labeled with the node itself belong to its parent's training set). Those samples are
        selected by index directly from `X`, so no per-node copy of the feature matrix is ever built.

        """
        columns = self.mlb.classes_ if self.mlb is not None else None
        rows_by_label = _rows_by_label(y, columns=columns)
        for node_id in dfs_preorder_nodes(self.graph_, self.root):
            progress.update(1)
            self._train_local_classifier(X, y, node_id, rows_by_label)

    @staticmethod
    def _training_rows(rows_by_label, below):
        """Sorted, de-duplicated indices of the samples labeled with any node in `below`."""
        groups = [rows_by_label[label] for label in below if label in rows_by_label]
        if not groups:
            return np.empty(0, dtype=np.intp)
        return np.unique(np.concatenate(groups))

    def _select_features(self, X, y):
        """
        Perform feature selection for the training data of a single node.

        Called once per trained node with that node's training samples `X` (a row subset of the
        training data, in the same format it was passed to `fit`) and their original targets `y`.
        Can be overridden by a sub-class to implement feature selection logic; the default is the
        identity. Nb. prediction passes unselected rows to the local classifiers, so an override
        must keep the number of columns (e.g. zero out unselected features rather than drop them).

        """
        return X

    def _build_metafeatures(self, y_rows):
        """
        Build the meta-features associated with a particular node.

        These are various features that can be used in training and prediction time,
        e.g the number of training samples available for the classifier trained at that node,
        the number of targets (classes) to be predicted at that node, etc.

        Parameters
        ----------
        y_rows : array-like, shape = [n_samples] or [n_samples, n_classes]
            The targets of the samples in the training set of the node.

        Returns
        -------
        metafeatures : dict
            Python dictionary of meta-features. The following meta-features are computed by default:
            * "n_samples" - Number of samples used to train classifier at given node.
            * "n_targets" - Number of targets (classes) to classify into at given node.

        """
        n_targets = nnz_columns_count(y_rows) if self.mlb is not None else len(np.unique(y_rows))
        return {"n_samples": y_rows.shape[0], "n_targets": n_targets}

    def _train_local_classifier(self, X, y, node_id, rows_by_label):
        is_leaf = self.graph_.out_degree(node_id) == 0
        if is_leaf and self.algorithm == "lcpn":
            # Leaf nodes do not get a classifier assigned in LCPN algorithm mode.
            self.logger.debug(
                "_train_local_classifier() - skipping leaf node %s when algorithm is 'lcpn'",
                node_id,
            )
            return

        child_of = children_by_descendant(self.graph_, node_id)
        subtree_rows = self._training_rows(rows_by_label, below=child_of)
        if not is_leaf:
            self.graph_.nodes[node_id][METAFEATURES] = self._build_metafeatures(y[subtree_rows])

        if self._is_inclusive and len(subtree_rows):
            # Every other document joins the training set as an all-negative row
            X_, y_ = self._select_features(X=X, y=y), self._inclusive_targets(y, subtree_rows, child_of)
        else:
            X_ = self._select_features(X=self._rows(X, subtree_rows), y=y[subtree_rows])
            X_, y_ = self._roll_up(X_, y[subtree_rows], child_of)
        if self._n_samples(X_) == 0:
            # No training data could be materialized for current node
            # TODO: support a "strict" mode flag to explicitly enable/disable fallback logic here?
            self.logger.warning(
                "_train_local_classifier() - not enough training data available to train, classification in branch will terminate at node %s",  # noqa:E501
                node_id,
            )
            return

        clf = self._local_classifier_for(node_id, y_)
        self.logger.debug(
            "_train_local_classifier() - training %s at node %s on %s samples",
            type(clf).__name__,
            node_id,
            len(y_),
        )
        clf.fit(X=X_, y=y_)
        self._check_local_classes(node_id, clf)
        self.graph_.nodes[node_id][CLASSIFIER] = clf
        if self.mlb is not None:
            # Children with no positive example here were not learned: a one-vs-rest estimator falls back to a
            # constant predictor for them (decision value 0), which a negative threshold would otherwise select
            self.graph_.nodes[node_id][TRAINED_CLASSES] = {self.mlb.classes_[j] for j in np.flatnonzero(y_.any(axis=0))}

    def _check_local_classes(self, node_id, clf):
        """A single-label local classifier must predict hierarchy nodes, or prediction cannot route."""
        if self.mlb is not None:
            return
        unknown = [class_ for class_ in clf.classes_ if class_ not in self._column_index()]
        if unknown:
            raise ValueError(
                f"Local classifier at node {node_id!r} predicts classes {unknown} that are not hierarchy nodes"
            )

    def _inclusive_targets(self, y, subtree_rows, child_of):
        """Indicator targets over every sample: the subtree's roll-up, zero rows elsewhere."""
        y_ = np.zeros((y.shape[0], len(self.mlb.classes_)), dtype=int)
        rolled_up = self.mlb.transform(rollup_targets(child_of, y[subtree_rows], mlb=self.mlb))
        y_[subtree_rows] = rolled_up.toarray() if issparse(rolled_up) else rolled_up
        return y_

    def _roll_up(self, X_, y_rows, child_of):
        """
        Turn each sample's target into the child (or children) of the current node it lies under.

        Returns the training data and targets for the node's local classifier: on a DAG a sample
        under several children is repeated once per child, so `X_` may grow.

        """
        if self.mlb is not None:
            return self._roll_up_multi_label(X_, y_rows, child_of)
        return self._roll_up_single_label(X_, y_rows, child_of)

    def _roll_up_single_label(self, X_, y_rows, child_of):
        labels, inverse = np.unique(y_rows, return_inverse=True)
        children = [child_of[label] for label in labels]
        if all(len(nodes) == 1 for nodes in children):
            # Every target lies under exactly one child (always the case on a tree): no expansion needed
            return X_, np.asarray([nodes[0] for nodes in children])[inverse]

        counts = np.fromiter((len(nodes) for nodes in children), dtype=np.intp)[inverse]
        positions = np.repeat(np.arange(len(y_rows)), counts)
        return self._rows(X_, positions), np.asarray(flatten_list(children[k] for k in inverse))

    def _roll_up_multi_label(self, X_, y_rows, child_of):
        y_ = self.mlb.transform(rollup_targets(child_of, y_rows, mlb=self.mlb))
        if issparse(y_):
            y_ = y_.toarray()
        # Drop samples whose rolled-up children are unknown to the binarizer (all-zero rows)
        keep = np.flatnonzero(y_.sum(axis=1) > 0)
        if len(keep) < y_.shape[0]:
            return self._rows(X_, keep), y_[keep]
        return X_, y_

    @property
    def _is_inclusive(self):
        return self.algorithm == "lcpn" and self.training_strategy == "inclusive"

    def _local_classifier_for(self, node_id, y_):
        """The estimator to fit at a node: the base estimator, or a constant predictor when the
        (single-label) training targets hold a single child."""
        if self.mlb is None and len(np.unique(y_)) == 1:
            # TODO: support a "strict" mode flag to explicitly enable/disable fallback logic here?
            self.logger.debug(
                "_train_local_classifier() - only a single target (child node) available to train classifier for node %s, Will trivially predict %s",  # noqa:E501
                node_id,
                y_[0],
            )
            # Nb. wrap as a 1-element array-like: DummyClassifier's parameter validation rejects numpy scalars
            return DummyClassifier(strategy="constant", constant=[y_[0]])
        return self._base_estimator_for(node_id)

    def _base_estimator_for(self, node_id):
        base_estimator = None
        if self.base_estimator is None:
            # No base estimator specified by user, try to pick best one
            base_estimator = self._make_base_estimator(node_id)

        elif isinstance(self.base_estimator, dict):
            # User provided dictionary mapping nodes to estimators
            if node_id in self.base_estimator:
                base_estimator = self.base_estimator[node_id]
            else:
                base_estimator = self.base_estimator[DEFAULT]

        elif is_estimator(self.base_estimator):
            # Single base estimator object, return a copy
            base_estimator = self.base_estimator

        else:
            # By default, treat as callable factory
            base_estimator = self.base_estimator(node_id=node_id, graph=self.graph_)

        return clone(base_estimator)

    def _make_base_estimator(self, node_id):
        """Create a default base estimator if a more specific one was not chosen by user."""
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )

    def _progress(self, total, desc, **kwargs):
        if self.progress_wrapper:
            return self.progress_wrapper(total=total, desc=desc)
        else:
            return DummyProgress()
