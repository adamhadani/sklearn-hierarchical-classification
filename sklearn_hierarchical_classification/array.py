"""Helpers for workings with sequences and (numpy) arrays."""

from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix, issparse


def flatten_list(lst):
    return list(chain(*lst))


def apply_along_rows(func, X):
    """
    Apply function row-wise to input matrix X.
    This will work for dense matrices (eg np.ndarray)
    as well as for CSR sparse matrices.

    """
    if issparse(X):
        # Nb. convert so that row slicing works for every sparse format (coo/dia/bsr are not subscriptable),
        # and slice rather than index so that the row stays 2-D for both sparse matrices and sparse arrays
        X = X.tocsr()
        return np.array([func(X[i : i + 1]) for i in range(X.shape[0])])
    else:
        # XXX might break vis-a-vis this issue merging: https://github.com/numpy/numpy/pull/8511
        # See discussion over issue with truncated string when using np.apply_along_axis here:
        #   https://github.com/numpy/numpy/issues/8352
        return np.ma.apply_along_axis(
            lambda x: func(x.reshape(1, -1)),
            axis=1,
            arr=X,
        )


def apply_rollup_Xy(X, y):
    """
    Parameters
    ----------
    X : (sparse) array-like, shape = [n_samples, n_features]
        Data.

    y : list-of-lists - [n_samples]
        For each sample, y maintains list of labels this sample should be used for in training.

    Returns
    -------
    X_, y_
        Transformed by 'flattening' out y parameter and duplicating corresponding rows in X

    """
    counts = np.fromiter((len(labelset) for labelset in y), dtype=np.intp, count=len(y))
    y_ = flatten_list(y)

    if np.all(counts == 1):
        # No expansion needed
        return X, y_

    if not isinstance(X, csr_matrix):
        # Row duplication is a single fancy-indexing operation on a CSR matrix
        X = csr_matrix(X)

    return X[np.repeat(np.arange(X.shape[0]), counts)], y_


def apply_rollup_Xy_raw(X, y):
    """
    Parameters
    ----------
    X : List

    y : list-of-lists - [n_samples]
        For each sample, y maintains list of labels this sample should be used for in training.

    Returns
    -------
    X_, y_
        Transformed by 'flattening' out y parameter and duplicating corresponding rows in X

    """
    if all(len(labelset) == 1 for labelset in y):
        # No expansion needed
        return X, flatten_list(y)

    # Our goal is to expand the equal labelsets into their own row within X
    # We do this by repeating each row exactly "labelset" times
    X_rows = []
    for x, labelset in zip(X, y, strict=True):
        X_rows.extend([x] * len(labelset))

    y_ = flatten_list(y)
    return X_rows, y_


def extract_rows_csr(matrix, rows):
    """
    Parameters
    ----------
    matrix : (sparse) csr_matrix

    rows : list of row ids

    Returns
    -------
    matrix_: (sparse) csr_matrix
        Same shape as `matrix`, with the desired rows kept and every other row zeroed

    """
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)

    keep = np.zeros(matrix.shape[0], dtype=bool)
    keep[np.asarray(rows, dtype=np.intp)] = True

    # Zero out the stored entries of every row not kept, then drop them from the structure
    matrix_ = matrix.copy()
    row_of_entry = np.repeat(np.arange(matrix_.shape[0]), np.diff(matrix_.indptr))
    matrix_.data[~keep[row_of_entry]] = 0
    matrix_.eliminate_zeros()
    return matrix_


def nnz_rows_ix(X):
    """Return row indices which have at least one non-zero column value."""
    return np.unique(X.nonzero()[0])


def nnz_columns_count(X):
    """Return count of columns which have at least one non-zero value."""
    return int(np.count_nonzero(np.count_nonzero(X, axis=0)))
