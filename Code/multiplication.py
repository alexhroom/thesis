"""
This script generates plots for the multiplication operator in two
bases, as well as the plot of the second-order relative spectrum.
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pyfilon import filon_fun_sin
import numpy as np

from specpol.ritz import ritz_bounded_L2, mult_sors
from specpol.plot import plot_ritz
from specpol.algebra import create_m_op
from specpol.common import generate_matrix


def plot_1(ritz_results, basis: str):
    """Plot a Ritz approximation.

    Parameters
    ----------
    ritz_results: Dict[int, np.array]
        A dictionary with keys corresponding to Ritz matrix size, and
        values corresponding to the eigenvalues of the matrix of that size.
    dbm: int or None, default None
        If not None, removes all datapoints with imaginary part smaller
        than `dbm`.

    Returns
    -------
    Figure, Axes, Axes
        Returns the figure and its two subplots for further modification if desired.
    """
    specs = ritz_results

    viridis = colormaps["viridis"].resampled(len(specs))

    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(1, 2, 1, adjustable="box")
    ax1.set_prop_cycle(color=viridis.colors)

    for i in specs:
        ax1.scatter([i] * len(specs[i]), specs[i].real, s=8)

    ax1.set_xlabel("size of Ritz matrix (number of rows/columns)")
    ax1.set_ylabel("real part of eigenvalues of the Ritz matrix")

    ax1.axhspan(0, 0.5, facecolor="green", alpha=0.2)
    ax1.axhspan(1, 1.5, facecolor="green", alpha=0.2)

    plt.savefig(f"mult_{basis}.png")


def plot_sors(ritz_results, sors):
    # here we modify the plot of the approximation to add bands
    # where we expect the spectrum to be

    fig, ax1, ax2 = plot_ritz(ritz_results)
    ax1.axhspan(0, 1/2, facecolor="green", alpha=0.2)
    ax1.axhspan(1, 1.5, facecolor="green", alpha=0.2)
    ax2.axvspan(0, 1/2, facecolor="green", alpha=0.2)
    ax2.axvspan(1, 1.5, facecolor="green", alpha=0.2)
    ax2.scatter(sors[0].real, sors[0].imag, alpha=0.3, color='blue')
    fig.suptitle("Ritz approximation and second-order relative spectrum")

    plt.savefig("mult_sors.png")


################### NUMERICS START HERE ########################################
# the symbol of our multiplication operator, m(x)
def step_slope(x):
    return x if x < 1 / 2 else x + 1 / 2

# `create_m_op` creates a multiplication operator with supplied symbol
step_operator = create_m_op(step_slope)

spec_step_slope = {}

for i in tqdm(range(50, 251, 25), desc="Approximating spectrum for exponentials..."):
    spec_step_slope[i] = np.linalg.eigvals(ritz_bounded_L2(step_operator, 1, i, 321))

plot_1(spec_step_slope, basis='exp')


# we define the script for sine from scratch
def ritz_bounded_L2_sin(
    operator, matrix_size: int, quad_mesh_size: int
) -> np.array:
    """Ritz approximation on L2(0, b) with sine functions.

    Parameters
    ----------
    operator: Operator
        The operator to approximate the spectrum of.
    b: float
        The upper limit of the domain.
    matrix_size: int
        The size of the square Ritz matrix.
    quad_mesh_size: int
        The size of the mesh used for quadrature.
        Must be an odd integer greater than 1.

    Returns
    -------
    np.array
        The Ritz matrix for the operator.
    """

    def onb_func(n: int):
        return lambda x: np.sin(np.pi*n*x)

    def integrand(n: int):
        return lambda x: 2 * operator(onb_func(n))(x)

    def entry_func(i: int, j: int) -> complex:
        """Calculate entry i,j of the Ritz matrix for the multiplication operator.

        These are the scalar products (M_f e_i, e_j)
        where e_k = 1/sqrt(2)sin(pi*n*x)
        """
        # the Filon quadrature has the second iexp as implicit
        return (
            filon_fun_sin(integrand(i), 0, 1, j * np.pi, quad_mesh_size)
        )

    ritz_matrix = generate_matrix(
        entry_func,
        matrix_size,
        start_index=0,
    )

    return ritz_matrix


for i in tqdm(range(50, 251, 25), desc="Approximating spectrum for sines..."):
    spec_step_slope[i] = np.linalg.eigvals(ritz_bounded_L2_sin(step_operator, 1, i, 321))

plot_1(spec_step_slope, basis='sin')

print("Calculating second-order relative spectrum...")
sors = mult_sors(step_slope, 1, 150, 321)

plot_sors(spec_step_slope, sors=sors)