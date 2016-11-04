from __future__ import print_function
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from stats.sk_old import rdivDemo
from utils import load_pickle
from glob import glob
from pdb import set_trace
from sklearn.metrics import auc
import numpy as np


def auc_temp(x, y):
    # auc0 = [np.sqrt(0.7*(1-xx)**2+0.3*(yy)**2) for xx, yy in zip(x,y)]
    # return auc0
    try:
        g_measure = [x * y /(x + y)]
        return np.mean(g_measure)  # , np.std(g_measure)
    except ZeroDivisionError:
        return 0
        # return np.mean(x), np.mean(y)


def print_sk_charts():
    files = glob(os.path.join(os.path.abspath("./"), '*.p'))
    for f in files:
        data = load_pickle(f)
        for key, value in data.iteritems():
            print("```\n## {}\n```\nname,Pd,Pf,G".format(key))
            stats = []
            for pd, pf in zip(value[0][0], value[0][1]):
                stats.append((pd[0], pd[1], pf[1], auc_temp(pd[1], pf[1])))
            # set_trace()
            stats = sorted(stats, key=lambda x: x[-1], reverse=True)

            for v in stats:
                print(v[0]+',','{0:.2f}, {1:.2f}, {2:0.2f}'.format(v[1], v[2], v[-1]))
                # set_trace()
    #
    return


if __name__ == "__main__":
    print_sk_charts()
