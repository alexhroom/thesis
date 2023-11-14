"""Tests the Filon quadrature algorithm."""

from specpol.common import filon_sin

import numpy as np
import pytest


def identity(val):
    return val


def quadratic(val):
    return val**2 + 3 * val + 7


def log(val):
    return np.log(1 + val)


def piecewise(val):
    if val < np.pi:
        return val
    else:
        return val + 1 / 2


def pi_sin(val):
    return np.sin(np.pi * val)


@pytest.mark.parametrize('points', [41, 81, 161, 321, 641])
@pytest.mark.parametrize(
    'function, exact',
    [
        (identity, (-np.pi / 5)),
        (quadratic, ((-3 * np.pi - 2 * np.pi**2) / 5)),
        (log, (-0.1961185699426520514276893141271915)),
        (piecewise, (-np.pi / 5)),
        (pi_sin, (10 * np.sin(2 * (np.pi**2)) / (np.pi**2 - 100))),
    ],
)
def test_filon_sin_points(points, function, exact):
    """Tests the accuracy of filon_sin over varying mesh sizes."""
    actual = filon_sin(function, (0, 2 * np.pi), points, 10)
    np.testing.assert_allclose(actual, exact, rtol=1e-2, atol=1e-2)
