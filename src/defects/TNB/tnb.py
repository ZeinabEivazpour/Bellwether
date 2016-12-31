from __future__ import print_function, division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src/defects')
if root not in sys.path:
    sys.path.append(root)

import warnings
from prediction.model import rf_model0
from py_weka.classifier import classify
from utils import *
from metrics.abcd import abcd
from metrics.recall_vs_loc import get_curve
from pdb import set_trace
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas
from plot.effort_plot import effort_plot


def target_details(test_set):
    set_trace()
    # return (lo, hi), mass


def get_test_mass(test_set):
    return


def get_distance(ranges):
    return


def weight_training(weights, training_instance):
    return


def tnb(source, target, n_rep=12):
    """
    TNB: Transfer Naive Bayes
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    result = dict()

    for tgt_name, tgt_path in target.iteritems():
        stats = []
        print("{}  \r".format(tgt_name[0].upper() + tgt_name[1:]))
        val = []
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:
                set_trace()
                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)
                pd, pf, g, auc = [], [], [], []
                min_max, test_mass = target_details(tgt)
                test_mass = get_test_mass(tgt)
                distance = get_distance(ranges=min_max)
                weighted_source = weight_training(distance, )
                set_trace()


def tnb_jur():
    from data.handler import get_all_projects
    all = get_all_projects()
    apache = all["Apache"]
    return tnb(apache, apache, n_rep=1)
    pass


if __name__ == "__main__":
    tnb_jur()
