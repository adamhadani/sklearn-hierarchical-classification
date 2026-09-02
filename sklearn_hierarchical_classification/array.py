"""Helpers for workings with sequences and (numpy) arrays."""

from itertools import chain

import numpy as np
from scipy.sparse import issparse


def flatten_list(lst):
    return list(chain(*lst))


def nnz_columns_count(X):
    """Return count of columns which have at least one non-zero value."""
    return int(np.count_nonzero(np.count_nonzero(X, axis=0)))


def apply_along_rows(func, X):
    """
    Apply function row-wise to input matrix X.
    This will work for dense matrices (eg np.ndarray)
    as well as for sparse matrices.

    """
    if issparse(X):
        # Nb. convert so that row slicing works for every sparse format (coo/dia/bsr are not subscriptable),
        # and slice rather than index so that the row stays 2-D for both sparse matrices and sparse arrays
        X = X.tocsr()
        return np.array([func(X[i : i + 1]) for i in range(X.shape[0])])
    return np.ma.apply_along_axis(lambda x: func(x.reshape(1, -1)), axis=1, arr=X)
