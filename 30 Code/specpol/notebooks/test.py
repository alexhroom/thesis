from typing import Dict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from specpol.ritz import ritz_sturm_liouville, ritz_unbounded_sturm_liouville
from specpol.plot import plot_ritz

import warnings
#warnings.simplefilter("error", np.ComplexWarning)

def potential(x):  # the potential Q(x) of the Sturm-Liouville operator
    return 0

rusl = dict()
for i in tqdm(range(50, 300, 50)):
    rusl[i] = ritz_unbounded_sturm_liouville(potential, i, 151)