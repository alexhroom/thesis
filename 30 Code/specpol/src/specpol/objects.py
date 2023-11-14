"""Classes for functional theoretic objects."""
from typing import Callable, TypeAlias

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
