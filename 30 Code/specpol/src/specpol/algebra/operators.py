"""Classes for functional theoretic objects."""
from typing import Callable, TypeAlias

import numpy as np
import scipy as sp

OperatorAction: TypeAlias = Callable[[Callable], Callable]


class Operator:
    """
    A class to represent an operator.

    Parameters
    ----------
    action: OperatorAction (Callable[[Callable], Callable])
        A function-valued function representing the application
        of the operator to
    """

    def __init__(self, action: OperatorAction):
        self.action = action

    def __call__(self, func):
        """
        Apply the operator with a function call.

        Parameters
        ----------
        func: Callable
            The Callable to which the operator is applied.
        """
        return self.action(func)


def create_m_op(func: Callable) -> Operator:
    """
    Creates the multiplication operator M_f with action
    (M_f u)(x) = f(x)u(x).

    Parameters
    ----------
    func: Callable
        The symbol of the multiplication operator.

    Returns
    -------
    Operator
        The multiplication operator with symbol `func`.
    """

    def mult_action(arg):
        return lambda x: func(x) * arg(x)

    return Operator(mult_action)


def create_m_perturb(func: Callable, lda: float) -> Operator:
    """
    Creates a rank-1 perturbation of the multiplication operator M_f
    with an extra eigenvalue at point `lda`.
    """
    # our rank-one perturbation operator is
    # Tu = Mu + (u, phi)phi
    # for some function phi satisfying the normalisation condition
    # $int_0^1 (|phi|^2 * (lda - m)) = 1$

    phi_unnorm = lambda x: x
    norm_func = lambda x: np.abs(phi_unnorm(x)) / (lda - func(x))
    phi = lambda x: phi_unnorm(x) / sp.integrate.quadrature(norm_func, 0, 1)[0]

    def m_perturb_action(arg):
        arg_times_phi = lambda x: arg(x) * phi(x)
        inner_prod_arg_phi = sp.integrate.quadrature(arg_times_phi, 0, 1)[0]

        return lambda x: func(x) * arg(x) + inner_prod_arg_phi * phi(x)

    return Operator(m_perturb_action)
