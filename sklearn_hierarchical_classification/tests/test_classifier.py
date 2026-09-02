"""
Unit-tests for the classifier interface.

"""

import pickle

import numpy as np
import pytest
from hamcrest import (
    assert_that,
    close_to,
    contains_inanyorder,
    equal_to,
    has_entries,
    has_item,
    is_,
)
from networkx import DiGraph, dfs_preorder_nodes
from numpy import where
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.estimator_checks import check_estimator

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import CLASSIFIER, DEFAULT, ROOT
from sklearn_hierarchical_classification.tests.fixtures import (
    make_classifier,
    make_classifier_and_data,
    make_clothing_graph,
    make_clothing_graph_and_data,
    make_digits_dataset,
    make_fruit_veg_raw_data,
    make_mlb_classifier_and_data_with_feature_extraction_pipeline,
)
from sklearn_hierarchical_classification.tests.matchers import matches_graph


RANDOM_STATE = 42


def test_estimator_inteface():
    """Run the scikit-learn estimator compatability test suite."""
    check_estimator(HierarchicalClassifier())


def test_fitted_attributes():
    """Test classifier attributes are set correctly after fitting."""
    n_classes = 10
    clf, (X, y) = make_classifier_and_data(n_classes=n_classes)

    clf.fit(X, y)

    assert_that(DiGraph(clf.class_hierarchy_), matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.graph_, matches_graph(DiGraph(clf.class_hierarchy)))
    assert_that(clf.classes_, contains_inanyorder(*range(n_classes)))
    assert_that(clf.n_classes_, is_(equal_to(n_classes)))
    assert_that(
        clf.graph_.nodes[ROOT],
        has_entries(
            metafeatures=has_entries(
                n_samples=X.shape[0],
                n_targets=n_classes,
            ),
        ),
    )


def test_trivial_hierarchy_classification():
    """Test that a trivial (degenerate) hierarchy behaves as expected."""
    clf, (X, y) = make_classifier_and_data(n_classes=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert_that(accuracy, is_(close_to(1.0, delta=0.05)))


@pytest.mark.slow
def test_mlb_hierarchy_classification_with_feature_extraction_pipeline():
    """Test multi-label classification with a feature extraction pipeline"""
    clf, (X, y) = make_mlb_classifier_and_data_with_feature_extraction_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_pred[where(y_pred == 0)] = -1
    accuracy = accuracy_score(y_test, y_pred > -0.2)

    assert_that(accuracy, is_(close_to(0.8, delta=0.05)))


def test_base_estimator_as_dict():
    """Test that specifying base_estimator as a dictionary mappings nodes to base estimators works."""
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": [1, 7],
        "B": [3, 8, 9],
    }
    clf = make_classifier(
        base_estimator={
            ROOT: KNeighborsClassifier(),
            "B": svm.SVC(),
            DEFAULT: MultinomialNB(),
        },
        class_hierarchy=class_hierarchy,
    )
    X, y = make_digits_dataset(
        targets=[1, 7, 3, 8, 9],
        as_str=False,
    )
    X_train, _X_test, y_train, _y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)

    assert_that(isinstance(clf.graph_.nodes[ROOT][CLASSIFIER], KNeighborsClassifier))
    assert_that(isinstance(clf.graph_.nodes["B"][CLASSIFIER], svm.SVC))
    assert_that(isinstance(clf.graph_.nodes["A"][CLASSIFIER], MultinomialNB))


def test_nontrivial_hierarchy_leaf_classification():
    r"""Test that a nontrivial hierarchy leaf classification behaves as expected
    under the default parameters.

    We build the following class hierarchy along with data from the handwritten digits dataset:

            <ROOT>
           /      \
          A        B
         / \      / \ \
        1   7    3   8  9

    """
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": [1, 7],
        "B": [3, 8, 9],
    }
    base_estimator = CalibratedClassifierCV(svm.SVC(gamma=0.001, kernel="rbf"), ensemble=False)
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
    )
    X, y = make_digits_dataset(
        targets=[1, 7, 3, 8, 9],
        as_str=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert_that(accuracy, is_(close_to(1.0, delta=0.02)))


def test_intermediate_node_training_data():
    r"""Test that a training set which includes intermediate (non-leaf) nodes
    as labels, as well as leaf nodes, constructs a correct classifier hierarchy

    """
    G, (X, y) = make_clothing_graph_and_data(root=ROOT)

    # Add a new node rendering "Bottoms" an intermediate node with training data
    G.add_edge("Bottoms", "Pants")

    assert_that(any(yi == "Pants" for yi in y), is_(False))
    assert_that(any(yi == "Bottoms" for yi in y), is_(True))

    base_estimator = LogisticRegression(
        solver="lbfgs",
        max_iter=1_000,
    )

    clf = HierarchicalClassifier(
        base_estimator,
        class_hierarchy=G,
        algorithm="lcpn",
        root=ROOT,
    )
    clf.fit(X, y)

    # Ensure non-terminal node with training data is included in its' parent classifier classes
    assert_that(clf.graph_.nodes()["Mens"]["classifier"].classes_, has_item("Bottoms"))


def test_nmlnp_strategy_with_float_stopping_criteria():
    # since NMLNP results in a mix of intermediate and leaf nodes,
    # make sure they are all of same dtype (str)
    class_hierarchy = {
        ROOT: ["A", "B"],
        "A": ["1", "5", "6", "7"],
        "B": ["2", "3", "4", "8", "9"],
    }
    base_estimator = CalibratedClassifierCV(svm.SVC(gamma=0.001, kernel="rbf"), ensemble=False)
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        prediction_depth="nmlnp",
        stopping_criteria=0.9,
    )

    X, y = make_digits_dataset()
    X_train, X_test, y_train, _y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert_that(list(y_pred), has_item("B"))


def test_nmlnp_strategy_on_tree_with_dummy_classifier():
    """Test classification works on a tree graph when one of the nodes has out-degree 1 resulting in
    creation of a "dummy" classifier at that node to trivially predict its child."""
    # since NMLNP results in a mix of intermediate and lefa nodes,
    # make sure they are all of same dtype (str)
    class_hierarchy = {
        ROOT: ["A", "B", "C"],
        "A": ["1", "5", "6", "7"],
        "B": ["2", "3", "8", "9"],
        "C": ["4"],
    }
    base_estimator = CalibratedClassifierCV(svm.SVC(gamma=0.001, kernel="rbf"), ensemble=False)
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        prediction_depth="nmlnp",
        stopping_criteria=0.9,
    )

    X, y = make_digits_dataset()
    X_train, X_test, y_train, _y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert_that(list(y_pred), has_item("4"))


def test_nmlnp_strategy_on_dag_with_dummy_classifier():
    """Test classification works on a "deep" DAG when one of the nodes has out-degree 1,
    resulting in creation of a "dummy" classifier at that node to trivially predict its child.

    This test case actually tests a few more subtle edge cases:

    - String-based target labels with length > 1
    - Multi-level degenerate sub-graphs, e.g some nodes having a sub-graph which is a path.

    """
    # since NMLNP results in a mix of intermediate and lefa nodes,
    # make sure they are all of same dtype (str)
    class_hierarchy = {
        ROOT: ["A", "B", "C"],
        "A": ["1", "5", "6", "7"],
        "B": ["2", "BC.1", "8", "9"],
        "BC.1": ["3a"],
        "C": ["BC.1"],
    }
    base_estimator = CalibratedClassifierCV(svm.SVC(gamma=0.001, kernel="rbf"), ensemble=False)
    clf = make_classifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        prediction_depth="nmlnp",
        stopping_criteria=0.9,
    )

    X, y = make_digits_dataset()
    y[where(y == "3")] = "3a"
    X_train, X_test, y_train, _y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert_that(list(y_pred), has_item("3a"))


def test_use_decision_function_with_preprocessed_features():
    """Test that a base estimator exposing only `decision_function` (e.g. LinearSVC) works in the
    default "preprocessed" mode, for both binary and multi-class local classifiers."""
    class_hierarchy = {
        ROOT: ["A", "B"],  # binary local classifier
        "A": [1, 7],
        "B": [3, 8, 9],  # multi-class local classifier
    }
    clf = make_classifier(
        base_estimator=svm.LinearSVC(),
        class_hierarchy=class_hierarchy,
        use_decision_function=True,
    )
    X, y = make_digits_dataset(targets=[1, 7, 3, 8, 9], as_str=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    assert_that(accuracy_score(y_test, y_pred), is_(close_to(1.0, delta=0.05)))
    assert_that(y_proba.shape, is_(equal_to((X_test.shape[0], clf.n_classes_))))


def test_raw_feature_extraction_with_predict_proba_pipeline():
    """Test "raw" mode with a feature extraction pipeline that exposes `predict_proba`,
    where samples outside a node's subtree must be dropped when training that node."""
    X, labels, class_hierarchy = make_fruit_veg_raw_data()
    y = np.array([leaf for leaf, _ in labels])
    clf = make_classifier(
        base_estimator=make_pipeline(CountVectorizer(), LogisticRegression()),
        class_hierarchy=class_hierarchy,
        feature_extraction="raw",
    )

    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    assert_that(list(y_pred), is_(equal_to(list(y))))
    assert_that(y_proba.shape, is_(equal_to((len(X), clf.n_classes_))))


class RecordingLogisticRegression(LogisticRegression):
    """Base estimator that keeps the training matrix it was fitted on, for inspection in tests."""

    def fit(self, X, y, sample_weight=None):
        self.X_fit_ = X
        return super().fit(X, y, sample_weight=sample_weight)


def test_dag_shared_descendant_rows_are_not_doubled():
    r"""On a DAG, rows of a leaf with two parents must reach the grandparent's classifier once,
    with their original values, not summed once per path.

            ROOT
           /    \
          A      B
         / \    / \
        M   L  L   N
    """
    graph = DiGraph([(ROOT, "A"), (ROOT, "B"), ("A", "L"), ("B", "L"), ("A", "M"), ("B", "N")])
    X = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]))
    y = np.array(["L", "L", "M", "N"])
    clf = make_classifier(base_estimator=RecordingLogisticRegression(), class_hierarchy=graph)

    clf.fit(X, y)

    X_root = clf.graph_.nodes[ROOT][CLASSIFIER].X_fit_
    assert_that(X_root.shape, is_(equal_to((6, 2))))  # the two L rows are used for both A and B
    assert_that(sorted(X_root.toarray().tolist()), is_(equal_to(sorted((X[[0, 1, 0, 1, 2, 3]]).toarray().tolist()))))


def test_fitted_model_does_not_retain_training_data():
    clf_small, (X_small, y_small) = make_classifier_and_data(n_classes=5, n_samples=200)
    clf_large, (X_large, y_large) = make_classifier_and_data(n_classes=5, n_samples=20_000)

    clf_small.fit(X_small, y_small)
    clf_large.fit(X_large, y_large)

    assert_that(any("X" in attrs for _, attrs in clf_large.graph_.nodes(data=True)), is_(False))
    size_small, size_large = len(pickle.dumps(clf_small)), len(pickle.dumps(clf_large))
    assert_that(
        size_large < 1.2 * size_small, is_(True), f"pickled size grew with n_samples: {size_small} -> {size_large}"
    )


def test_select_features_hook_receives_each_node_training_subset():
    """The overridable _select_features hook is called once per trained node with that node's
    training rows (and their targets), so subclasses can do per-node feature selection."""
    calls = []

    class SelectingClassifier(HierarchicalClassifier):
        def _select_features(self, X, y):
            calls.append((X.shape[0], len(np.unique(y))))
            return X

    class_hierarchy = {ROOT: ["A", "B"], "A": [1, 7], "B": [3, 8, 9]}
    X, y = make_digits_dataset(targets=[1, 7, 3, 8, 9], as_str=False)
    clf = SelectingClassifier(base_estimator=LogisticRegression(max_iter=1_000), class_hierarchy=class_hierarchy)

    clf.fit(X, y)

    expected = {
        (X.shape[0], 5),  # ROOT: all rows, 5 leaf targets
        (np.isin(y, [1, 7]).sum(), 2),  # A
        (np.isin(y, [3, 8, 9]).sum(), 3),  # B
    }
    assert_that(set(calls), is_(equal_to(expected)))
    assert_that(len(calls), is_(equal_to(3)))


def test_local_classifiers_are_trained_in_depth_first_order():
    """Training order is the deterministic depth-first order of the hierarchy (successor insertion
    order), so base estimators drawing from a seeded global RNG give reproducible fits."""
    graph = make_clothing_graph()
    graph.add_edge("Bottoms", "Pants")
    graph.add_edge("Bottoms", "Shorts")
    visited = []

    class OrderRecordingClassifier(HierarchicalClassifier):
        def _train_local_classifier(self, X, y, node_id, rows_by_label):
            visited.append(node_id)
            return super()._train_local_classifier(X, y, node_id, rows_by_label)

    X = np.random.normal(size=(60, 4))
    y = np.random.choice(["Shirts", "Jackets", "Swim", "Pants", "Shorts"], size=60)
    clf = OrderRecordingClassifier(base_estimator=LogisticRegression(), class_hierarchy=graph)

    clf.fit(X, y)

    assert_that(visited, is_(equal_to(list(dfs_preorder_nodes(graph, ROOT)))))


def test_dense_input_stays_dense_at_dag_nodes():
    """A base estimator that requires dense input must get dense rows at every node, including DAG
    nodes where rows are duplicated across children."""
    graph = DiGraph([(ROOT, "A"), (ROOT, "B"), ("A", "L"), ("B", "L"), ("A", "M"), ("B", "N")])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = np.array(["L", "M", "N", "L"] * 10)
    clf = make_classifier(base_estimator=GaussianNB(), class_hierarchy=graph)

    clf.fit(X, y)

    assert_that(set(clf.predict(X).tolist()) <= {"L", "M", "N"}, is_(True))


def test_mlb_children_unknown_to_binarizer_leave_node_untrained():
    """If a node's children are not classes of the MultiLabelBinarizer, its rolled-up targets are
    all-zero rows: there is nothing to train on, so the node gets no classifier (and a warning)."""
    X, labels, class_hierarchy = make_fruit_veg_raw_data()
    leaves_only = MultiLabelBinarizer().fit([[leaf] for leaf, _ in labels])
    clf = make_classifier(
        base_estimator=make_pipeline(CountVectorizer(), OneVsRestClassifier(LogisticRegression())),
        class_hierarchy=class_hierarchy,
        feature_extraction="raw",
        mlb=leaves_only,
    )

    with pytest.warns(UserWarning):
        clf.fit(X, leaves_only.transform([[leaf] for leaf, _ in labels]))

    assert_that(CLASSIFIER in clf.graph_.nodes[ROOT], is_(False))
    assert_that(CLASSIFIER in clf.graph_.nodes["fruit"], is_(True))


def test_mlb_accepts_sparse_indicator_targets():
    X, labels, class_hierarchy = make_fruit_veg_raw_data()
    mlb = MultiLabelBinarizer(sparse_output=True).fit(labels)
    clf = make_classifier(
        base_estimator=make_pipeline(CountVectorizer(), OneVsRestClassifier(LogisticRegression())),
        class_hierarchy=class_hierarchy,
        feature_extraction="raw",
        mlb=mlb,
    )

    clf.fit(X, mlb.transform(labels))

    assert_that(clf.predict_proba(X).shape, is_(equal_to((len(X), len(mlb.classes_)))))
