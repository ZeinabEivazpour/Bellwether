from __future__ import print_function, division

import numpy as np
import pandas as pd

from utils import list2dataframe
from pdb import set_trace

def get_curve(test_set, distribution, loc_name="$loc"):
    if isinstance(test_set, pd.core.frame.DataFrame):
        dframe = test_set
    elif isinstance(test_set, list):
        dframe = list2dataframe(test_set)

    loc = dframe[loc_name]

    set_trace()

    for K in np.arange(0.1, 1):
        predicted = [1 if val > K else 0 for val in distribution]

        pass
    return
