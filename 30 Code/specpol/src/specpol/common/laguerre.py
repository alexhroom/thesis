"""Generalised Laguerre polynomials."""
from functools import lru_cache

@lru_cache(maxsize=None)
def laguerre(n, a, x):
    if n == 0:
        return 1
    if n == 1:
        return -x + a + 1
    if n == 2:
        return x**2/2 - (a+2)*x + (a+1)*(a+2)/2
    else:
        return (2 + (a-1-x)/n)*laguerre(n-1, a, x) - (1 + (a-1)/n)*laguerre(n-2, a, x)