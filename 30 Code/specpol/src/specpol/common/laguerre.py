"""Generalised Laguerre polynomials."""
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.optimize import fsolve


@lru_cache(maxsize=None)
def laguerre(n: int, a: float, x: complex) -> float:
    """Calculate the associated Laguerre polynomial L_n^a(x).

    Parameters
    ----------
        n: int
            the order of the Laguerre polynomial.
        a: float
            the generalised alpha-parameter of the Laguerre polynomial.
        x: complex
            the value at which the polynomial is evaluated.

    Returns
    -------
    float
        The Laguerre polynomial L_n^a evaluated at x.
    """
    if n == 0:
        return 1
    if n == 1:
        return -x + a + 1
    if n == 2:
        return x**2 / 2 - (a + 2) * x + (a + 1) * (a + 2) / 2
    return (2 + (a - 1 - x) / n) * laguerre(n - 1, a, x) - (1 + (a - 1) / n) * laguerre(n - 2, a, x)


def lagquad(mesh_size: int) -> Tuple[np.array, np.array]:
    r"""Calculate the sample points and weights for Gauss-Laguerre quadrature.

    These sample points $x_i$ and weights $w_i$ approximate an integral like so:
    $$\int_0^\infty f(x) exp(-x)dx \approx \sum_{i=0}^n w_i * f(x_i).$$

    Parameters
    ----------
    mesh_size: int
        The number of sample points and weights to calculate.

    Returns
    -------
    Tuple[np.array, np.array]
    Two arrays; one of the sample points, one of the weights.
    """

    def objective(x):
        if isinstance(x, float):
            return laguerre(mesh_size, 0, x)
        if isinstance(x, np.array):
            return np.array([laguerre(mesh_size, 0, xi) for xi in x])

    def fprime(x):
        return laguerre(mesh_size-1, 1, x)

    # calculate roots of L_n
    roots = fsolve(objective, 0, fprime=fprime)

    weights = [x/((mesh_size+1)**2 * laguerre(mesh_size+1, 0, x)**2) for x in roots]



