"""Ritz method for various types of operator."""
from typing import Callable, Tuple

import numpy as np
from pyfilon import filon_fun_sin

from specpol.common import generate_matrix
from specpol.objects import Operator


def multiplication_operator_ritz(func: Callable, matrix_size: int, quad_mesh_size: int) -> np.array:
    """
    Approximate the spectrum of a multiplication operator M_f on L^2(0, 1)
    via the Ritz method, where f is a real-valued function.

    Parameters
    ----------
    func: Callable
        The symbol f of the operator M_f.
    matrix_size: int
        The size of the square Ritz matrix.
    quad_mesh_size: int
        The size of the mesh used for quadrature.
        Must be an odd integer greater than 1.

    Returns
    -------
    np.array
        A one-dimensional array listing the estimated spectral points of the operator.
    """

    # we have a factor of 2 in the integrand to account for the
    # sqrt(2) attached to each sin(n pi x)
    def integrand(n):
        return lambda x: 2 * func(x) * np.sin(n * np.pi * x)

    def entry_func(i, j):
        """
        Function to define entries of the Ritz matrix
        for the multiplication operator.

        These are the scalar products (M_f e_i, e_j)
        where e_k = sin(k*pi*x).
        """
        # the Filon quadrature has the second sin as implicit
        return filon_fun_sin(integrand(i), (0, 1), quad_mesh_size, j * np.pi)

    ritz_matrix = generate_matrix(entry_func, (matrix_size, matrix_size))

    return np.linalg.eigvals(ritz_matrix)


# pylint: disable=invalid-name
def ritz_L2(
    operator: Operator,
    onb: Callable,
    interval: Tuple[float, float],
    matrix_size: int,
    quad_mesh_size: int,
) -> np.array:
    """
    Estimate the spectrum of an operator on L2 with the Ritz (truncation) method.

    Parameters
    ----------
    operator: Operator
        The operator for which we are estimating the spectrum.
    onb: Callable
        The definition of the sequence used as an orthonormal
        basis for this spectrum.
    interval: Tuple[float, float]
        The interval on which the function space is defined.
    matrix_size: int
        The size of the square Ritz matrix.
    quad_mesh_size: int
        The size of the mesh used for quadrature.
        Must be an odd integer greater than 1.

    Returns
    -------
    np.array
        A one-dimensional array listing the estimated spectral points of the operator.
    """

    raise NotImplementedError
