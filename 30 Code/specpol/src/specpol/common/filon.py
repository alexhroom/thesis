"""Filon quadrature for numerical integration."""
from typing import Tuple, Callable

import numpy as np


def filon_sin(
    func: Callable, interval: Tuple[float, float], num_points: int, sin_coeff: float
) -> float:
    """
    Filon quadrature rule for `func` multiplied by sin(Mx) over the interval [0, 1].
    Adapted from QUADPACK.

    Citation:
    R. Piessens, E. De Doncker-Kapenga and C. W. Überhuber.
    QUADPACK: a subroutine package for automatic integration.
    Springer, ISBN: 3-540-12553-1. 1983.

    Parameters
    ----------
    func: Callable
        The function f(x) to integrate.
    interval: Tuple(float, float)
        The interval to integrate over.
    num_points: int
        The number of datapoints at which the function is evaluated, including the
        endpoints. Must be an odd number greater than 1.
    sin_coeff: float
        The coefficient of the sine function in the integrand; M in 'sin(Mx)'

    Returns
    -------
    float
        The quadrature estimate of the integral func(x)*sin(m pi x) over the interval [0, 1].
    """
    if (num_points % 2 == 0) or (num_points < 2):
        err = "num_points must be an odd integer greater than 1."
        raise RuntimeError(err)

    h = (interval[1] - interval[0]) / (num_points - 1)

    # calculation of endpoints
    values = np.linspace(interval[0], interval[1], num_points)
    theta = sin_coeff * h

    # 1/6 is chosen as the cutoff for using power series rather than
    # closed form based on the hardware spec of the ILLIAC II...
    # is there a better way now? probably
    if abs(theta) <= 1 / 6:
        alpha, beta, gamma = calculate_coeffs_powerseries(theta)
    else:
        alpha, beta, gamma = calculate_coeffs_closedform(theta)

    # discretise the function
    ftab = [func(x) for x in values]

    s2n = sum((ftab[x] * np.sin(sin_coeff * values[x])) for x in range(0, num_points, 2)) - 0.5 * (
        ftab[-1] * np.sin(sin_coeff * values[-1]) + ftab[0] * np.sin(sin_coeff * values[0])
    )
    s2nm1 = sum((ftab[x] * np.sin(sin_coeff * values[x])) for x in range(1, num_points - 1, 2))

    output = h * (
        (
            alpha
            * (ftab[0] * np.cos(sin_coeff * values[0]) - ftab[-1] * np.cos(sin_coeff * values[-1]))
        )
        + beta * s2n
        + gamma * s2nm1
    )

    return output


def calculate_coeffs_powerseries(theta: float) -> Tuple[float, float, float]:
    """
    Calculate the quadrature coefficients alpha, beta, gamma using a power series.

    Parameters
    ----------
    theta: float
        The angle used.

    Returns
    -------
    Tuple(float, float, float)
        Alpha, beta, and gamma respectively.
    """

    def ps_theta(coefficients):
        """
        Return the power series evaluated at theta with given coefficients.
        """
        return sum(coeff * (theta**n) for n, coeff in enumerate(coefficients))

    alpha = ps_theta([0, 0, 0, 2 / 45, 0, -2 / 315, 0, 2 / 4725])
    beta = ps_theta([2 / 3, 0, 2 / 15, 0, -4 / 105, 0, 2 / 567, 0, -4 / 22275])
    gamma = ps_theta([4 / 3, 0, -2 / 15, 0, 1 / 210, 0, -1 / 11340])

    return alpha, beta, gamma


def calculate_coeffs_closedform(theta: float) -> Tuple[float, float, float]:
    """
    Calculate the quadrature coefficients alpha, beta, gamma exactly.

    Parameters
    ----------
    theta: float
        The angle used.

    Returns
    -------
    Tuple(float, float, float)
        Alpha, beta, and gamma respectively.
    """

    sint = np.sin(theta)
    cost = np.cos(theta)

    alpha = (theta**2 + theta * sint * cost - 2 * (sint**2)) / theta**3
    beta = (2 * theta + 2 * theta * (cost**2) - 4 * sint * cost) / theta**3
    gamma = 4 * (sint - (theta * cost)) / theta**3

    return alpha, beta, gamma
