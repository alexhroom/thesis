"""Utilities for generating and manipulating matrices."""
from typing import Callable, Tuple

import numpy as np


def generate_matrix(entry_func: Callable, shape: Tuple[int, int], start_index: int = 1) -> np.array:
    """
    Generate a matrix from a function f(i, j), where
    f(i, j) is the value of the entry in row i, column j.

    Parameters
    ----------
    entry_func: Callable
        The function which evaluates the matrix at each entry.
    shape: Tuple[int, int]
        A pair of integers containing the number of rows and
        columns in the output matrix, respectively.
    start_index: int
        The number to start indexing matrix entries from. Defaults to 1,
        so the first entry is (1, 1)

    Returns
    -------
    np.array
        The matrix with entries A_{i,j} = f(i, j).
    """
    index_matrix = np.indices(shape, dtype=float) + start_index
    vectorised_func = np.vectorize(entry_func)

    return vectorised_func(*index_matrix)
