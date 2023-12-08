"""Ritz methods for Sturm-Liouville operators."""
from functools import lru_cache
from typing import Callable

import numpy as np
from pyfilon import filon_fun_iexp
from scipy.special import roots_laguerre

from specpol.common import generate_matrix, laguerre


def ritz_sturm_liouville(
    potential: Callable,
    domain_len: float,
    matrix_size: int,
    quad_mesh_size: int,
    *,
    dbm: bool = False,
) -> np.array:
    """Approximate the spectrum of a Sturm-Liouville operator with potential Q.

    The Sturm-Liouville operator is defined
    Ly = -y'' + Q(x)y

    Parameters
    ----------
    potential: Callable
        The function Q in the operator definition.
    domain_len: float
        The length of the domain (0, `domain_len`)
    matrix_size: int
        The size of the Ritz matrix used for approximation.
    quad_mesh_size: int
        The number of interpolation points used for quadrature.
    dbm: bool, default False
        Whether to add a dissipative barrier to the operator.
    """

    def onb_func(n: int) -> Callable:
        return lambda x: 1 / np.sqrt(domain_len) * np.exp(2j * np.pi * n * x / domain_len)

    @lru_cache(maxsize=matrix_size)
    def integrand(n: int) -> Callable:
        def bvp(x: complex) -> complex:
            if x == 0:
                return (-2j * np.pi * n * x / domain_len * np.sin(np.pi / 8)) / np.cos(
                    np.pi / 8,
                )
            if x == domain_len:
                return 0
            return (
                (4 * np.pi**2 * n**2) / (domain_len**2) * onb_func(n)(x)
                + potential(x) * onb_func(n)(x)
                + 1j * (dbm is True and x < 80) * onb_func(n)(x)
            )

        return bvp

    def entry_func(i: int, j: int) -> complex:
        """Calculate entry i, j of the Ritz matrix for the operator."""
        # the Filon quadrature has the second iexp as implicit
        return (
            1
            / np.sqrt(domain_len)
            * filon_fun_iexp(
                integrand(i),
                0,
                domain_len,
                -2 * j * np.pi / domain_len,
                quad_mesh_size,
            )
        )

    ritz_matrix = generate_matrix(
        entry_func,
        matrix_size,
        start_index=0,
        doubleinf=True,
    )

    return np.linalg.eigvals(ritz_matrix)


def ritz_unbounded_sturm_liouville(
    potential: Callable,
    matrix_size: int,
    quad_mesh_size: int,
    *,
    dbm: bool = False,
) -> np.array:
    r"""Ritz method for a Sturm-Liouville operator on the half-line [0, \infty).

    Parameters
    ----------
    potential: Callable
        The function Q in the operator definition.
    matrix_size: int
        The size of the Ritz matrix used for approximation.
    quad_mesh_size: int
        The number of interpolation points used for quadrature.
    dbm: bool, default False
        Whether to add a dissipative barrier to the operator.
    """

    # the weighted Laguerre polynomials L_n * exp(-x/2) form
    # an orthonormal basis for the half-line
    @lru_cache(maxsize=matrix_size)
    def integrand(n: int) -> Callable:
        def bvp(x: float) -> complex:
            # with boundary condition y(0) = y'(0)*tan(pi/8)
            # y'(0) = power-1 term in y(x) in this case
            if x == 0:
                return laguerre(n - 1, 1, 0) * np.tan(np.pi / 8)
            # we factor out the exponential term as the quadrature will
            # implicitly re-apply it as a weight
            # the action of the Sturm-Liouville operator on the basis is
            # (Ln'' - Ln' + Ln/4 + QLn)*exp(-x/2)
            return (-(0 if n <= 1 else laguerre(n - 2, 2, x))
                - (0 if n==0 else laguerre(n - 1, 1, x))
                - laguerre(n, 0, x) / 4
                + potential(x) * laguerre(n, 0, x)
                + 1j * (dbm is True and x < matrix_size/2) * laguerre(n, 0, x)
            )

        return bvp

    sample_points, weights = roots_laguerre(quad_mesh_size)

    def entry_func(i: int, j: int) -> float:
        return sum(
            np.array([integrand(i)(x) * laguerre(j, 0, x) for x in sample_points]) * weights,
        )

    ritz_matrix = generate_matrix(entry_func, matrix_size, start_index=0)

    return np.linalg.eigvals(ritz_matrix)
