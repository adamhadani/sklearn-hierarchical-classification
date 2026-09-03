"""Helpers for workings with sequences and (numpy) arrays."""

from collections.abc import Iterable
from itertools import chain
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


T = TypeVar("T")


def flatten_list(lst: Iterable[Iterable[T]]) -> list[T]:
    return list(chain(*lst))


def nnz_columns_count(X: ArrayLike) -> int:
    """Return count of columns which have at least one non-zero value."""
    return int(np.count_nonzero(np.count_nonzero(X, axis=0)))


def top_k_mask(scores: NDArray[Any], k: int) -> NDArray[np.bool_]:
    """Boolean mask of each row's `k` highest-scoring columns, ties broken by column order (all False when `k` is 0)."""
    n_rows, n_columns = scores.shape
    mask = np.zeros((n_rows, n_columns), dtype=bool)
    k = min(k, n_columns)
    if k > 0:
        best = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        mask[np.arange(n_rows)[:, None], best] = True
    return mask
