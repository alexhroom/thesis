"""
This script generates the plot for the perturbed multiplication operator
with and without a dissipative barrier.
"""
from specpol.ritz import ptb_ritz
from specpol.plot import plot_ritz

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def step_slope(x):  # the symbol of our multiplication operator
    return x if x < 1 / 2 else x + 1 / 2


def plot(ritz_results, dbm=False):
    # we add some extra lines and spans to our plots
    # to show where the 'actual' spectrum is
    fig, ax1, ax2 = plot_ritz(ritz_results)

    ax1.set_ylim(0, 1.5)

    ax1.axhspan(0, 0.5, facecolor="green", alpha=0.2)
    ax1.axhspan(1, 1.5, facecolor="green", alpha=0.2)
    ax1.axhline(0.7, linestyle="dotted", alpha=0.3)

    ax2.set_ylim(-0.25, 1.5)
    ax2.set_xlim(0, 1.5)
    ax2.axvline(0.7, linestyle="dotted", alpha=0.3)

    plt.savefig(f"perturbed_mult_{'dbm' if dbm else ''}.png", bbox_inches="tight")


ptb_specs = dict()
ptb_specs_dbm = dict()
for i in tqdm(range(50, 251, 25), desc="Approximating spectra..."):
    ptb_specs[i] = np.linalg.eigvals(ptb_ritz(step_slope, i, 161))
    ptb_specs_dbm[i] = np.linalg.eigvals(ptb_ritz(step_slope, i, 161, dbm=True))

plot(ptb_specs)
plot(ptb_specs_dbm, dbm=True)
