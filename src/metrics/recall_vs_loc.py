from __future__ import print_function, division

import numpy as np
import pandas as pd
from pdb import set_trace
from utils import list2dataframe


def get_curve(loc, actual, predicted, loc_name="$loc", cutoff=0.60):

    sorted_loc = np.array(loc)[np.argsort(loc)]
    sorted_act = np.array(actual)[np.argsort(loc)]

    # predicted = np.array([1 if val > cutoff else 0 for val in distribution])
    sorted_prd = np.array(predicted)[np.argsort(loc)]
    recall, loc = [], []
    tp, fn, Pd = 0, 0, 0
    loc_nos = set(sorted_loc)
    for a, p, l in zip(sorted_act, sorted_prd, sorted_loc):
            tp += 1 if a == 1 and p == 1 else 0
            fn += 1 if a == 1 and p == 0 else 0
            Pd = tp / (tp + fn) if (tp + fn) > 0 else Pd
            # print(tp, fn, Pd)
            loc.append(l)
            recall.append(int(Pd * 100))
    # set_trace()
    return recall, loc
