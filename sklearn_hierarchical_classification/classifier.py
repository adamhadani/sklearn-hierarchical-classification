"""
Hierarchical classifier interface.

"""

from itertools import chain

import numpy as np
from networkx import DiGraph, descendants, is_tree
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

from sklearn_hierarchical_classification.array import (
    apply_along_rows,
    apply_rollup_Xy,
    apply_rollup_Xy_raw,
    nnz_columns_count,
)
from sklearn_hierarchical_classification.constants import (
    CLASSIFIER,
    DEFAULT,
    METAFEATURES,
    ROOT,
)
from sklearn_hierarchical_classification.decorators import logger
from sklearn_hierarchical_classification.dummy import DummyProgress
from sklearn_hierarchical_classification.graph import make_flat_hierarchy, rollup_nodes
from sklearn_hierarchical_classification.validation import is_estimator, validate_parameters


def _rows_by_label(y):
    """Map each distinct label in `y` to the sorted array of row indices carrying it."""
    order = np.argsort(y, kind="stable")
    labels, starts = np.unique(y[order], return_index=True)
    bounds = np.append(starts, len(order))
    return {label: np.sort(order[start:end]) for label, start, end in zip(labels, bounds[:-1], bounds[1:], strict=True)}


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
        This parameter is used when the "algorithm" parameter is to set to "lcn", and dictates how training data
        is constructed for training the binary classifier at each node.

    stopping_criteria: function, float, or None.
        This parameter is used when the "prediction_depth" parameter is set to "nmlnp", and is used to evaluate
        at a given node whether classification should terminate or continue further down the hierarchy.

        When set to a float, the prediction will stop if the reported confidence at current classifier is below
        the provided value.

        When set to a function, the callback function will be called with the current node attributes,
        including its metafeatures, and the current classification results.
        This allows the user to define arbitrary logic that can decide whether classification should stop at
        the current node or continue. The function should return True if classification should continue,
        or False if classification should stop at current node.

    root : integer, string
        The unique identifier for the qualified root node in the class hierarchy. The hierarchical classifier
        assumes that the given class hierarchy graph is a rooted DAG, e.g has a single designated root node
        of in-degree 0. This node is associated with a special identifier which defaults to a framework provided one,
        but can be overridden by user in some cases, e.g if the original taxonomy is already rooted and there"s no need
        for injecting an artifical root node.

    progress_wrapper : progress generator or None
        If value is set, will attempt to use the given generator to display progress updates. This added functionality
        is especially useful within interactive environments (e.g in a testing harness or a Jupyter notebook). Setting
        this value will also enable verbose logging. Common values in tqdm are `tqdm_notebook` or `tqdm`

    feature_extraction : "preprocessed", "raw"
        Determines the feature extraction policy the classifier uses.
        When set to "raw", the classifier will expect the raw training examples are passed in to `.fit()` and `.train()`
        as X. This means that the base_estimator should point to a sklearn Pipeline that includes feature extraction.
        When set to "preprocessed", the classifier will expect X to be a pre-computed feature (sparse) matrix.

    mlb : MultiLabelBinarizer or None
        For multi-label classification, the MultiLabelBinarizer instance that was used for creating the y variable.

    mlb_prediction_threshold : float
        For multi-label prediction tasks (when `mlb` is set to a MultiLabelBinarizer instance), can define a prediction
        score threshold to use for considering a label to be a prediction. Defaults to zero.

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
            Multi-class targets. A binary indicator matrix (as produced by the `MultiLabelBinarizer`
            passed as `mlb`) turns on multi-label classification; this is only supported together
            with `feature_extraction="raw"`.

        Returns
        -------
        self

        """
        if self.feature_extraction == "raw":
            # In raw mode, only validate targets (y) format and
            # that targets and training data (X) are of same cardinality, since
            # X will in general not be a 2D feature matrix, but rather the raw training examples,
            # e.g. text snippets or images.
            y = check_array(
                y,
                accept_sparse="csr",
                ensure_all_finite=True,
                ensure_2d=False,
                dtype=None,
            )
            if len(X) != y.shape[0]:
                raise ValueError("bad input shape: len(X) != y.shape[0]")
        else:
            X, y = validate_data(self, X, y, accept_sparse="csr")

        check_classification_targets(y)

        # Check that parameter assignment is consistent
        self._check_parameters()

        # Initialize NetworkX Graph from input class hierarchy
        self.class_hierarchy_ = self.class_hierarchy or make_flat_hierarchy(list(np.unique(y)), root=self.root)
        self.graph_ = DiGraph(self.class_hierarchy_)
        self.is_tree_ = is_tree(self.graph_)
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
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes].
            Predicted multi-class targets.

        """
        check_is_fitted(self, "graph_")

        def _classify(x):
            path, _ = self._recursive_predict(x, root=self.root)
            if self.mlb:
                return path
            else:
                return path[-1]

        if self.feature_extraction == "raw":
            return np.array([_classify(X[i]) for i in range(len(X))])
        else:
            X = validate_data(self, X, accept_sparse="csr", reset=False)

        y_pred = apply_along_rows(_classify, X=X)
        return y_pred

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        check_is_fitted(self, "graph_")

        def _classify(x):
            _, scores = self._recursive_predict(x, root=self.root)
            return scores

        if self.feature_extraction == "raw":
            return np.array([_classify(X[i]) for i in range(len(X))])
        else:
            X = validate_data(self, X, accept_sparse="csr", reset=False)

        y_pred = apply_along_rows(_classify, X=X)
        return y_pred

    @property
    def n_classes_(self):
        return len(self.classes_)

    def _check_parameters(self):
        """Check the parameter assignment is valid and internally consistent."""
        validate_parameters(self)

    def _train_local_classifiers(self, X, y, progress):
        """Train the local classifier at every node reachable from the root.

        Each node's training set is the set of samples labeled with a strict descendant of that node
        (samples labeled with the node itself belong to its parent's training set). Those samples are
        selected by index directly from `X`, so no per-node copy of the feature matrix is ever built.

        """
        rows_by_label = None if self.mlb is not None else _rows_by_label(y)
        for node_id in chain([self.root], descendants(self.graph_, self.root)):
            progress.update(1)
            self._train_local_classifier(X, y, node_id, rows_by_label=rows_by_label)

    def _training_rows(self, y, node_id, rows_by_label):
        """Sorted indices of the samples labeled with a strict descendant of `node_id`."""
        below = descendants(self.graph_, node_id)
        if self.mlb is None:
            groups = [rows_by_label[label] for label in below if label in rows_by_label]
            return np.sort(np.concatenate(groups)) if groups else np.empty(0, dtype=np.intp)

        # Multi-label: y is a binary indicator matrix whose columns follow mlb.classes_
        columns = [column for column, label in enumerate(self.mlb.classes_) if label in below]
        if not columns:
            return np.empty(0, dtype=np.intp)
        return np.flatnonzero(np.asarray(y[:, columns].sum(axis=1)).ravel() > 0)

    def _select_features(self, X, y):
        """
        Perform feature selection for the training data of a single node.

        Called once per trained node with that node's training samples `X` (a row subset of the
        training data, in the same format it was passed to `fit`) and their original targets `y`.
        Can be overridden by a sub-class to implement feature selection logic; the default is the
        identity.

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

        rows = self._training_rows(y, node_id, rows_by_label)
        y_rows = y[rows]
        if self.mlb is not None and issparse(y_rows):
            y_rows = y_rows.toarray()
        if not is_leaf:
            self.graph_.nodes[node_id][METAFEATURES] = self._build_metafeatures(y_rows)

        if len(rows) == 0:
            # No training data could be materialized for current node
            # TODO: support a "strict" mode flag to explicitly enable/disable fallback logic here?
            self.logger.warning(
                "_train_local_classifier() - not enough training data available to train, classification in branch will terminate at node %s",  # noqa:E501
                node_id,
            )
            return

        X_ = [X[i] for i in rows] if self.feature_extraction == "raw" else X[rows]
        X_ = self._select_features(X=X_, y=y_rows)

        # Roll every sample's target up to the child (or children, on a DAG) of this node it lies under
        y_rolled_up = rollup_nodes(graph=self.graph_, source=node_id, targets=y_rows, mlb=self.mlb)
        if self.mlb is not None:
            y_ = self.mlb.transform(y_rolled_up)
        elif self.feature_extraction == "raw":
            X_, y_ = apply_rollup_Xy_raw(X_, y_rolled_up)
        else:
            X_, y_ = apply_rollup_Xy(X_, y_rolled_up)

        num_targets = len(np.unique(y_))
        self.logger.debug(
            "_train_local_classifier() - Training local classifier for node: %s, n_samples: %s, len(y): %s, n_targets: %s",  # noqa:E501
            node_id,
            len(rows),
            len(y_),
            num_targets,
        )

        if num_targets == 1:
            # Training data could be materialized for only a single target at current node
            # TODO: support a "strict" mode flag to explicitly enable/disable fallback logic here?
            constant = y_[0]
            self.logger.debug(
                "_train_local_classifier() - only a single target (child node) available to train classifier for node %s, Will trivially predict %s",  # noqa:E501
                node_id,
                constant,
            )

            # Nb. wrap as a 1-element array-like: DummyClassifier's parameter validation rejects numpy scalars
            clf = DummyClassifier(strategy="constant", constant=[constant])
        else:
            clf = self._base_estimator_for(node_id)

        clf.fit(X=X_, y=y_)
        self.graph_.nodes[node_id][CLASSIFIER] = clf

    def _local_scores(self, clf, x):
        """
        Score a single sample with the local classifier at a node.

        Returns a 1-D array of per-class scores aligned with ``clf.classes_``, regardless of
        feature extraction mode (a raw sample is wrapped as a length-1 batch) and of whether
        scores come from ``decision_function`` or ``predict_proba``.

        """
        x_ = [x] if self.feature_extraction == "raw" else x
        if self.use_decision_function and hasattr(clf, "decision_function"):
            scores = np.asarray(clf.decision_function(x_)).reshape(-1)
            if len(clf.classes_) == 2 and scores.shape[0] == 1:
                # A binary decision_function returns a single signed score for classes_[1]
                scores = np.array([-scores[0], scores[0]])
            return scores
        return np.asarray(clf.predict_proba(x_)).reshape(-1)

    def _recursive_predict(self, x, root):
        if CLASSIFIER not in self.graph_.nodes[root]:
            return None, None

        clf = self.graph_.nodes[root][CLASSIFIER]
        path = [root]
        path_proba = []
        class_proba = np.zeros_like(self.classes_, dtype=np.float64)

        while clf:
            probs = self._local_scores(clf, x)
            argmax = np.argmax(probs)
            score = probs[argmax]

            path_proba.append(score)
            if self.mlb is not None:
                predictions = []

            # Report probabilities in terms of complete class hierarchy
            if len(clf.classes_) == 1:
                prediction = clf.classes_[0]

            for local_class_idx, class_ in enumerate(clf.classes_):
                if self.mlb:
                    # when we have a multi-label binarizer
                    class_idx = class_
                    class_proba[class_idx] = probs[local_class_idx]
                    if class_proba[class_idx] > self.mlb_prediction_threshold:
                        predictions.append(self.mlb.classes_[class_])
                else:
                    try:
                        class_idx = self.classes_.index(class_)
                    except ValueError:
                        # This may happen if the classes_ enumeration we construct during fit()
                        # has a mismatch with the individual node classifiers" classes_.
                        self.logger.error(
                            "Could not find index in self.classes_ for class_ = '%s' (type: %s). path: %s",
                            class_,
                            type(class_),
                            path,
                        )
                        raise
                    class_proba[class_idx] = probs[local_class_idx]
                    if local_class_idx == argmax:
                        prediction = class_

            if self.mlb is None:
                if self._should_early_terminate(
                    current_node=path[-1],
                    prediction=prediction,
                    score=score,
                ):
                    break

                # Update current path
                path.append(prediction)
                clf = self.graph_.nodes[prediction].get(CLASSIFIER, None)
            else:
                clf = None
                for prediction in predictions:
                    pred_path, preds_prob = self._recursive_predict(x, prediction)
                    path.append(prediction)
                    if preds_prob is not None:
                        class_proba += preds_prob
                        path.extend(pred_path)

        return path, class_proba

    def _should_early_terminate(self, current_node, prediction, score):
        """
        Evaluate whether classification should terminate at given step.

        This depends on whether early-termination, as dictated by the the "prediction_depth"
          and "stopping_criteria" parameters, is triggered.

        """
        if self.prediction_depth != "nmlnp":
            # Prediction depth parameter does not allow for early termination
            return False

        if isinstance(self.stopping_criteria, float) and score < self.stopping_criteria:
            if current_node == self.root:
                return False

            self.logger.debug(
                "_should_early_terminate() - score %s < %s, terminating at node %s",
                score,
                self.stopping_criteria,
                current_node,
            )
            return True
        elif callable(self.stopping_criteria):
            return self.stopping_criteria(
                current_node=self.graph_.nodes[current_node],
                prediction=prediction,
                score=score,
            )

        return False

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
