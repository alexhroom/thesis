"""
This script generates all plots for the perturbed periodic Sturm-Liouville
operator from Aceto, Ghelardoni and Marletta (2006).
This includes the graphs of eigenfunctions.
"""

from typing import Dict

import numpy as np
import mpmath as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colormaps

from specpol.ritz import sturm_liouville_bdd, sturm_liouville_halfline


def plot_ritz(ritz_results: Dict[int, np.array], *, dbm: int | None = None, iv: str):
    """Plot a Ritz approximation.

    Parameters
    ----------
    ritz_results: Dict[int, np.array]
        A dictionary with keys corresponding to Ritz matrix size, and
        values corresponding to the eigenvalues of the matrix of that size.
    dbm: int or None, default None
        If not None, removes all datapoints with imaginary part smaller
        than `dbm`.
    iv: str
        Independent variable for axis labels

    Returns
    -------
    Figure, Axes, Axes
        Returns the figure and its two subplots for further modification if desired.
    """
    if dbm is not None:
        specs = {
            key: np.array([v for v in ritz_results[key] if v.imag > dbm])
            for key in ritz_results
        }
    else:
        specs = ritz_results

    viridis = colormaps["viridis"].resampled(len(specs))

    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(1, 2, 1, adjustable="box")
    ax1.set_prop_cycle(color=viridis.colors)

    for i in specs:
        ax1.scatter([i] * len(specs[i]), specs[i].real, s=8)

    ax1.set_xlabel(iv)
    ax1.set_ylabel("real part of eigenvalues of the Ritz matrix")

    ax2 = fig.add_subplot(1, 2, 2, adjustable="box")
    ax2.set_prop_cycle(color=viridis.colors)
    ax2.set_xlabel("real part of eigenvalues of the Ritz matrix")
    ax2.set_ylabel("imaginary part of eigenvalues of the Ritz matrix")

    for i in specs:
        ax2.scatter(specs[i].real, specs[i].imag, s=8)

    norm = plt.Normalize(min(specs), max(specs))
    cmap = plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax2, ticks=list(specs.keys()))
    cb.set_label(
        iv,
        rotation=270,
        labelpad=15,
    )

    return fig, ax1, ax2


def plot(ritz_results: Dict[int, np.array], title="", dbm=None, iv="", notes=""):
    # plot the approximation with some added bars and lines

    fig, ax1, ax2 = plot_ritz(ritz_results, dbm=dbm, iv=iv)
    fig.suptitle(title, fontsize=12)

    ax1.set_ylim(-0.5, 2.5)

    ax1.axhline(0.335936534279424, linestyle="--", alpha=0.6)
    ax1.axhline(0.536620364148446, linestyle="--", alpha=0.6)
    ax1.axhline(0.580834838005921, linestyle="--", alpha=0.6)
    ax1.axhline(0.591500609480355, linestyle="--", alpha=0.6)

    ax1.axhline(0.949634991713441, linestyle="--", alpha=0.6)
    ax1.axhline(1.24466406161563, linestyle="--", alpha=0.6)
    ax1.axhline(1.29192807845892, linestyle="--", alpha=0.6)

    ax1.axhspan(-0.3784, -0.34766, facecolor="green", alpha=0.2)
    ax1.axhspan(0.594799, 0.9180581788, facecolor="green", alpha=0.2)
    ax1.axhspan(1.2931662851, 2.2851569481, facecolor="green", alpha=0.2)

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.5, 1.25)

    ax2.axvline(0.335936534279424, linestyle="--", alpha=0.6)
    ax2.axvline(0.536620364148446, linestyle="--", alpha=0.6)
    ax2.axvline(0.580834838005921, linestyle="--", alpha=0.6)
    ax2.axvline(0.591500609480355, linestyle="--", alpha=0.6)

    ax2.axvline(0.949634991713441, linestyle="--", alpha=0.6)
    ax2.axvline(1.24466406161563, linestyle="--", alpha=0.6)
    ax2.axvline(1.29192807845892, linestyle="--", alpha=0.6)

    ax2.axvspan(-0.3784, -0.34766, facecolor="green", alpha=0.2)
    ax2.axvspan(0.594799, 0.9180581788, facecolor="green", alpha=0.2)
    ax2.axvspan(1.2931662851, 2.2851569481, facecolor="green", alpha=0.2)

    # get lowercase first word of title and iv for
    tis = title.casefold().split(" ", 1)[0]
    ivs = iv.casefold().split(" ", 1)[0]
    plt.savefig(f"aceto_{tis}_{ivs}_{notes}.png", bbox_inches="tight")


def plot_eigfn(eigfn, title="", typ=""):
    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot()

    ax1.plot(
        np.linspace(0, 500, 1500),
        [complex(eigfn(x)).real for x in np.linspace(0, 500, 1500)],
        color="green",
    )
    ax1.set_title(title)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("Value of eigenfunction $\phi(x)$")

    plt.savefig(f"eigfn_{typ}.png", bbox_inches="tight")


###### NUMERICS START HERE #####################################################
# strings for independent variables (used in plotting)
mat_size = "size of Ritz matrix (number of rows/columns)"
coup = "coupling constant $\gamma$"
length = "barrier length $R$"


def potential(x):  # the potential Q(x) of the Sturm-Liouville operator
    return np.sin(x) - 40 / (1 + x**2)


sl_spec = {}
sl_spec_dbm = {}

for i in tqdm(range(50, 121, 10), desc="Approximating truncated operator..."):
    sl_spec[i] = sturm_liouville_bdd(
        potential, (0, 70 * np.pi), i, 321, (np.pi / 8, np.pi / 2)
    )
    sl_spec_dbm[i] = sturm_liouville_bdd(
        potential,
        (0, 70 * np.pi),
        i,
        321,
        (np.pi / 8, np.pi / 2),
        dbm=(lambda x: (x <= 150)),
    )

plot(
    sl_spec,
    title="Truncated Galerkin approximation of Sturm-Liouville operator",
    iv=mat_size,
)
plot(
    sl_spec_dbm,
    title="Truncated Galerkin approximation of Sturm-Liouville operator, dissipative barrier applied",
    iv=mat_size,
    notes="dbm",
)


rusl_mat = {}
rusl_len = {}
rusl_coup = {}

for i in tqdm(
    range(50, 121, 10),
    desc="Approximating non-truncated operator with varying matrix size...",
):
    rusl_mat[i] = sturm_liouville_halfline(
        potential, i, 250, alpha=np.pi / 8, dbm=(lambda x: (x <= 100))
    )
for i in tqdm(
    range(50, 151, 10),
    desc="Approximating non-truncated operator with varying barrier length...",
):
    rusl_len[i] = sturm_liouville_halfline(
        potential, 100, 250, alpha=np.pi / 8, dbm=(lambda x: (x <= i))
    )
for i in tqdm(
    np.linspace(0, 1, 25),
    desc="Approximating non-truncated operator with varying coupling constant...",
):
    rusl_coup[i] = sturm_liouville_halfline(
        potential, 100, 250, alpha=np.pi / 8, dbm=(lambda x: i * (x <= 100))
    )

plot(
    rusl_mat,
    title="Galerkin approximation of Sturm-Liouville operator on half-line, dissipative barrier applied",
    iv=mat_size,
)
plot(
    rusl_len,
    title="Galerkin approximation of Sturm-Liouville operator on half-line, dissipative barrier applied",
    dbm=0.95,
    iv=length,
)
plot(
    rusl_coup,
    title="Galerkin approximation of Sturm-Liouville operator on half-line, dissipative barrier applied",
    iv=coup,
)

# approximate and save eigenfunctions
rusl_eigpairs = sturm_liouville_halfline(
    potential, 120, 250, alpha=np.pi / 8, dbm=(lambda x: x <= 100), returns="vectors"
)


def disc_eigfunc(vec):
    return lambda x: sum(
        weight * mp.laguerre(i, 0, x) * np.exp(-x / 2)
        for i, weight in enumerate(vec, start=0)
    )


safe_val = rusl_eigpairs.filter(
    lambda x: x.real < 0.5 and x.real > 0.25 and x.imag > 0.95
)
sval, svec = list(safe_val.data.items())[0]
seigfn = disc_eigfunc(svec)
plot_eigfn(seigfn, title="Eigenfunction corresponding to a true eigenvalue", typ="true")

poll_val = rusl_eigpairs.filter(lambda x: x.real < 0.25 and x.real > 0)
pval, pvec = list(poll_val.data.items())[0]
peigfn = disc_eigfunc(pvec)
plot_eigfn(
    peigfn, title="Eigenfunction corresponding to a polluting eigenvalue", typ="poll"
)
