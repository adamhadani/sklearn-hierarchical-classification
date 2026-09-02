"""Helpers for workings with sequences and (numpy) arrays."""

from itertools import chain

import numpy as np


def flatten_list(lst):
    return list(chain(*lst))


def nnz_columns_count(X):
    """Return count of columns which have at least one non-zero value."""
    return int(np.count_nonzero(np.count_nonzero(X, axis=0)))
