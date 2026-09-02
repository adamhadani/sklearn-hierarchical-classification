import numpy as np
from hamcrest import assert_that, equal_to, is_

from sklearn_hierarchical_classification.array import flatten_list, nnz_columns_count


def test_flatten_list():
    assert_that(flatten_list([[0, 1], [], [2]]), is_(equal_to([0, 1, 2])))


def test_nnz_columns_count():
    X = np.array([[1, 0, 0, 0], [0, 0, 2, 0]])

    assert_that(nnz_columns_count(X), is_(equal_to(2)))
