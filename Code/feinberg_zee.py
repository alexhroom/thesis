"""
This script plots the Feinberg-Zee Random Hopping Model example.
"""
import matplotlib.pyplot as plt
from matplotlib import colormaps
import random
import numpy as np

from specpol.ritz import ritz_tridiag, supercell


def plot_ritz(spec):
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot()
    ax1.scatter(spec.real, spec.imag, c="#440154", s=8)
    ax1.set_title(
        f"Feinberg-Zee Galerkin approximation for $\sigma={sigma}, p={sigma_prob}, n=2000$"
    )
    ax1.set_xlabel("Real part of Ritz matrix eigenvalue")
    ax1.set_ylabel("Imaginary part of Ritz matrix eigenvalue")

    plt.savefig("feinberg_zee_ritz.png")


def plot_supercell(spec):
    fig = plt.figure(dpi=300)
    ax1 = fig.add_subplot()
    viridis = colormaps["twilight_shifted"].resampled(len(spec))
    ax1.set_prop_cycle(color=viridis.colors)
    for theta_vals in spec.values():
        ax1.scatter(theta_vals.real, theta_vals.imag, s=8, alpha=0.1)
    ax1.set_title(
        f"Feinberg-Zee supercell approximation for $\sigma={sigma}, p={sigma_prob}, n=2000$"
    )
    ax1.set_xlabel("Real part of Ritz matrix eigenvalue")
    ax1.set_ylabel("Imaginary part of Ritz matrix eigenvalue")
    norm = plt.Normalize(0, 2 * np.pi)
    cmap = plt.get_cmap("twilight_shifted")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax1, ticks=list(np.linspace(0, 2 * np.pi, 10)))
    cb.set_label(
        r"value of $\theta$",
        rotation=270,
        labelpad=15,
    )

    plt.savefig("feinberg_zee_supercell.png")


###### NUMERICS START HERE #####################################################
sigma_prob = 0.5
sigma = 0.9025
random.seed(1862)  # use same random seed for all approximations


def subdiag(n):
    p = random.random()
    if p < sigma_prob:
        return sigma
    return -sigma


# regular Ritz method
spec = ritz_tridiag(subdiag, 0, 1, 2000)

plot_ritz(spec)

# supercell method
cell_spec = supercell(subdiag, 0, 1, 2000)

plot_supercell(cell_spec)
