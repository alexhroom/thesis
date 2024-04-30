"""
This script generates the plot for the almost Mathieu (Ten Martini) operator.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import colormaps

from specpol.ritz import ritz_tridiag, supercell

def plot_ritz_tridiag(spec):
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot()
    for alpha, vals in spec.items():
        ax1.scatter(vals.real, [alpha]*len(vals), c = "#440154" ,s = 8)
    ax1.set_title("Almost Mathieu Galerkin approximation, $n = 750$")
    ax1.set_xlabel("Real part of Ritz matrix eigenvalue")
    ax1.set_ylabel(r"$\alpha$")

    plt.savefig("ten_martini_ritz.png", bbox_inches='tight')

def plot_floquet_bloch(cell_spec):
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot()
    for a in cell_spec.keys():
        viridis = colormaps["twilight_shifted"].resampled(len(cell_spec[a]))
        ax1.set_prop_cycle(color=viridis.colors)
        for theta_vals in cell_spec[a].values():
            ax1.scatter(theta_vals.real, [a]*len(theta_vals), s = 8, alpha=0.1)
    ax1.set_title("Almost Mathieu operator Floquet-Bloch calculation over one period")
    ax1.set_xlabel("Real part of Ritz matrix eigenvalue")
    ax1.set_ylabel(r"$\alpha$")
    norm = plt.Normalize(0, 2*np.pi)
    cmap = plt.get_cmap("twilight_shifted")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax1, ticks=list(np.linspace(0, 2*np.pi, 10)))
    cb.set_label(
        r"value of $\theta$",
        rotation=270,
        labelpad=15,
    )

    plt.savefig("ten_martini_floquet.png", bbox_inches='tight')


######## NUMERICS START HERE ###################################################
lda = 1
theta = 1
def diag(alpha):
    """The diagonal of the Jacobi matrix."""
    return lambda n: 2 * lda * np.cos(2*np.pi*(theta + n*alpha))


# regular Ritz method
spec = {}
for a in tqdm(np.linspace(-1, 1, 201), ):
    spec[a] = ritz_tridiag(1, diag(a), 1, 750)

plot_ritz_tridiag(spec)


# Floquet-Bloch method
cell_spec = {}
for a in tqdm(np.linspace(-1, 1, 201)):
    cell_spec[a] = supercell(1, diag(a), 1, 100, alpha_samples=10)

plot_floquet_bloch(cell_spec)