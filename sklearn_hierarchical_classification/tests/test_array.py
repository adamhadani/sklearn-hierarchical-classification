import numpy as np
from hamcrest import assert_that, equal_to, is_
from scipy.sparse import coo_matrix, csr_array

from sklearn_hierarchical_classification.array import (
    apply_along_rows,
    apply_rollup_Xy,
    apply_rollup_Xy_raw,
    nnz_columns_count,
)


def test_apply_rollup_xy():
    X = np.arange(9).reshape(3, 3)
    y_rolled_up = [
        [0, 1],
        [2],
        [3, 4, 5],
    ]

    X_, y_ = apply_rollup_Xy(X, y_rolled_up)

    assert_that((X_[0] != X_[1]).nnz, is_(equal_to(0)))
    assert_that((X_[3] != X_[4]).nnz, is_(equal_to(0)))
    assert_that((X_[4] != X_[5]).nnz, is_(equal_to(0)))

    for i in range(6):
        assert_that(y_[i], is_(equal_to(i)))


def test_apply_rollup_xy_raw():
    X = ["doc a", "doc b", "doc c"]
    y_rolled_up = [
        [0, 1],
        [2],
        [3, 4, 5],
    ]

    X_, y_ = apply_rollup_Xy_raw(X, y_rolled_up)

    assert_that(X_, is_(equal_to(["doc a", "doc a", "doc b", "doc c", "doc c", "doc c"])))
    assert_that(y_, is_(equal_to(list(range(6)))))


def test_apply_rollup_xy_raw_no_expansion():
    X = ["doc a", "doc b"]
    y_rolled_up = [[0], [1]]

    X_, y_ = apply_rollup_Xy_raw(X, y_rolled_up)

    assert_that(X_, is_(equal_to(X)))
    assert_that(y_, is_(equal_to([0, 1])))


def test_apply_along_rows_sparse_formats():
    X = np.arange(6).reshape(3, 2)
    expected = [1, 5, 9]

    for X_sparse in (csr_array(X), coo_matrix(X)):
        assert_that(list(apply_along_rows(lambda row: row.sum(), X_sparse)), is_(equal_to(expected)))


def test_apply_rollup_xy_all_empty():
    X = np.arange(6).reshape(3, 2)

    X_, y_ = apply_rollup_Xy(X, [[], [], []])

    assert_that(X_.shape, is_(equal_to((0, 2))))
    assert_that(y_, is_(equal_to([])))


def test_nnz_columns_count():
    X = np.array([[1, 0, 0, 0], [0, 0, 2, 0]])

    assert_that(nnz_columns_count(X), is_(equal_to(2)))
