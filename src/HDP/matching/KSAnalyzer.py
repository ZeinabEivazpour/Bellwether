from __future__ import division, print_function
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)
from pdb import set_trace
import pandas as pd
from utils import *
from scipy.stats import ks_2samp
import networkx as nx

def KSAnalyzer(source, target, cutoff=0.05):
    ## temp -- refactor
    source = source[source.columns[3:-1]]  # Refactor!
    target = target[target.columns[3:-1]]  # Refactor!
    matches = dict()
    for col_name_src, col_name_tgt in pairs(source, target):

        _, p_val = ks_2samp(source[col_name_src],
                            target[col_name_tgt])
        print("{}\t{}\t{}".format(col_name_src, col_name_tgt, p_val))

    set_trace()


def weightedBipartite():
    pass


def _test__KSAnalyzer():
    data0 = pd.read_csv(os.path.join(root, "data/Jureczko/ant/ant-1.6.csv"))
    data1 = pd.read_csv(os.path.join(root, "data/Jureczko/ant/ant-1.7.csv"))
    KSAnalyzer(source=data0, target=data1, cutoff=0.05)
    # ----- Debug -----
    set_trace()


def _test__weightedBipartite():
    pass


if __name__ == "__main__":
    # set_trace()
    _test__KSAnalyzer()
