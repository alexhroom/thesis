"""Filon quadrature for numerical integration."""
from typing import Tuple, Callable

import numpy as np
from numba import njit


def filon_sin(
    func: Callable[[float], float], interval: Tuple[float, float], num_points: int, sin_coeff: float
) -> float:
    """
    Filon quadrature rule for `func` multiplied by sin(mx) over an interval.
    Adapted from FILON: https://people.math.sc.edu/Burkardt/cpp_src/filon/filon.html

    Reference:
    Stephen Chase, Lloyd Fosdick,
    An Algorithm for Filon Quadrature,
    Communications of the Association for Computing Machinery,
    Volume 12, Number 8, August 1969, pages 453-457.

    Stephen Chase, Lloyd Fosdick,
    Algorithm 353: Filon Quadrature,
    Communications of the Association for Computing Machinery,
    Volume 12, Number 8, August 1969, pages 457-458.

    Bo Einarsson,
    Algorithm 418: Calculation of Fourier Integrals,
    Communications of the ACM,
    Volume 15, Number 1, January 1972, pages 47-48.


    Parameters
    ----------
    func: Callable, float -> float
        The function f(x) over which to integrate.
    interval: Tuple[float, float]
        The interval to integrate over.
    num_points: int
        The number of datapoints at which the function is evaluated, including the
        endpoints. Must be an odd integer greater than 1.
    sin_coeff: float
        The coefficient of the sine function in the integrand; `m` in 'sin(mx)'

    Returns
    -------
    float
        The quadrature estimate of the integral func(x)*sin(mx) over the interval given.
    """
    ftab = np.array([func(x) for x in np.linspace(interval[0], interval[1], num_points)])

    return filon_sin_tab(ftab, interval, num_points, sin_coeff)


# pylint: disable=too-many-locals, consider-using-generator
# cannot use generator as Numba doesn't support it
@njit
def filon_sin_tab(
    ftab: np.array, interval: Tuple[float, float], num_points: int, sin_coeff: float
) -> float:
    """
    Filon quadrature rule for an already-discretised function
    multiplied by sin(mx) over an interval.

    Parameters
    ----------
    ftab: np.array
        The discretised function over the mesh to integrate.
    interval: Tuple[float, float]
        The interval to integrate over.
    num_points: int
        The number of datapoints at which the function is evaluated, including the
        endpoints. Must be an odd integer greater than 1.
    sin_coeff: float
        The coefficient of the sine function in the integrand; `m` in 'sin(mx)'

    Returns
    -------
    float
        The quadrature estimate of the integral func(x)*sin(mx) over the interval given.
    """
    if (num_points % 2 == 0) or (num_points < 2):
        raise RuntimeError("num_points must be an odd integer greater than 1.")

    h = (interval[1] - interval[0]) / (num_points - 1)

    # create mesh and calculate theta
    values = np.linspace(interval[0], interval[1], num_points)
    theta = sin_coeff * h

    # 1/6 is chosen as the cutoff for using power series rather than
    # closed form based on the hardware spec of the ILLIAC II...
    # is there a better way now? probably
    if abs(theta) <= 1 / 6:  # use power series for small angles to get greater accuracy

        def ps_theta(coefficients, theta):
            """
            Return the power series evaluated at theta with given coefficients.
            """
            return sum([coeff * (theta**n) for n, coeff in enumerate(coefficients)])

        alpha = ps_theta([0, 0, 0, 2 / 45, 0, -2 / 315, 0, 2 / 4725], theta)
        beta = ps_theta([2 / 3, 0, 2 / 15, 0, -4 / 105, 0, 2 / 567, 0, -4 / 22275], theta)
        gamma = ps_theta([4 / 3, 0, -2 / 15, 0, 1 / 210, 0, -1 / 11340], theta)

    else:  # else, use closed form
        sint = np.sin(theta)
        cost = np.cos(theta)

        alpha = (theta**2 + theta * sint * cost - 2 * (sint**2)) / theta**3
        beta = (2 * theta + 2 * theta * (cost**2) - 4 * sint * cost) / theta**3
        gamma = 4 * (sint - (theta * cost)) / theta**3

    s2n = sum(
        [(ftab[x] * np.sin(sin_coeff * values[x])) for x in range(0, num_points, 2)]
    ) - 0.5 * (ftab[-1] * np.sin(sin_coeff * values[-1]) + ftab[0] * np.sin(sin_coeff * values[0]))
    s2nm1 = sum([(ftab[x] * np.sin(sin_coeff * values[x])) for x in range(1, num_points - 1, 2)])

    output = h * (
        (
            alpha
            * (ftab[0] * np.cos(sin_coeff * values[0]) - ftab[-1] * np.cos(sin_coeff * values[-1]))
        )
        + beta * s2n
        + gamma * s2nm1
    )

    return output
