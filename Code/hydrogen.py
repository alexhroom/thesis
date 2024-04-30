"""
This script generates the eigenvalues and plot for the hydrogen atom
Schrodinger equation.
"""
from typing import Dict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps

from specpol.ritz import sturm_liouville_halfline


def plot(ritz_results: Dict[int, np.array], dbm=None):
    # plot the approximation with some added bars and lines
    if dbm is not None:
        specs = {
            key: np.array([v for v in ritz_results[key] if v.imag > dbm]) for key in ritz_results
        }
    else:
        specs = ritz_results

    viridis = colormaps["viridis"].resampled(len(specs))

    fig = plt.figure(dpi=300)

    ax1 = fig.add_subplot()
    ax1.set_prop_cycle(color=viridis.colors)

    for i in specs:
        ax1.scatter([i] * len(specs[i]), specs[i].real, s=8)

    ax1.set_xlabel("size of Ritz matrix (number of rows/columns)")
    ax1.set_ylabel("real part of eigenvalues of the Ritz matrix")

    ax1.set_ylim(-0.1, 0)
    ax1.axhline(-1/16, linestyle="--", alpha=0.6)
    ax1.axhline(-1/36, linestyle="--", alpha=0.6)
    ax1.axhline(-1/64, linestyle="--", alpha=0.6)
    ax1.axhline(-1/100, linestyle="--", alpha=0.6)
    ax1.axhline(-1/144, linestyle="--", alpha=0.6)
    ax1.axhline(-1/196, linestyle="--", alpha=0.6)

    plt.savefig("hydrogen.png")


def potential(x):  # the potential Q(x) of the Sturm-Liouville operator
    return -1/x + 2/(x**2)


rusl = {}
vc = {}
for i in tqdm(range(50, 175, 25), desc="Approximating spectrum..."):
    rusl[i] = sturm_liouville_halfline(potential, i, 200, np.pi/2)

plot(rusl)