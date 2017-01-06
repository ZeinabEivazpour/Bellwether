from __future__ import print_function, division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src/smells')
if root not in sys.path:
    sys.path.append(root)

import warnings
from prediction.model import logistic_model, rf_model
from py_weka.classifier import classify
from utils import *
from metrics.abcd import abcd
from metrics.recall_vs_loc import get_curve
from pdb import set_trace
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas
from plot.effort_plot import effort_plot
from tabulate import tabulate
from datasets.handler2 import get_all_datasets


def weight_training(test_instance, training_instance):
    head = training_instance.columns
    new_train = training_instance[head[:-1]]
    new_train = (new_train - test_instance[head[:-1]].mean()) / test_instance[head[:-1]].std()
    new_train[head[-1]] = training_instance[head[-1]]
    new_train.dropna(axis=1, inplace=True)
    tgt = new_train.columns
    new_test = (test_instance[tgt[:-1]] - test_instance[tgt[:-1]].mean()) / (
        test_instance[tgt[:-1]].std())

    new_test[tgt[-1]] = test_instance[tgt[-1]]
    new_test.dropna(axis=1, inplace=True)
    columns = list(set(tgt[:-1]).intersection(new_test.columns[:-1]))+[tgt[-1]]
    return new_train[columns], new_test[columns]



def predict_defects(train, test):
    actual = test[test.columns[-1]].values.tolist()
    actual = [1 if act == "T" else 0 for act in actual]
    # Binarize data
    # set_trace()
    # predicted, distr = nbayes(train, test)
    predicted, distr = rf_model(train, test)
    return actual, predicted, distr


def bellw(source, target, n_rep=12):
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
        charts = []
        print("{} \r".format(tgt_name[0].upper() + tgt_name[1:]))
        val = []
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:

                src = pandas.read_csv(src_path)
                tgt = pandas.read_csv(tgt_path)

                pd, pf, g, auc = [], [], [], []
                for _ in xrange(n_rep):
                    _train, __test = weight_training(test_instance=tgt, training_instance=src)
                    # set_trace()
                    actual, predicted, distribution = predict_defects(train=_train, test=__test)

                    # loc = tgt["$loc"].values
                    # loc = loc * 100 / np.max(loc)
                    # recall, loc, au_roc = get_curve(loc, actual, predicted, distribution)
                    # effort_plot(recall, loc,
                    #             save_dest=os.path.abspath(os.path.join(root, "plot", "plots", tgt_name)),
                    #             save_name=src_name)

                    p_d, p_f, p_r, rc, f_1, e_d, _g, auroc = abcd(actual, predicted, distribution)
                    set_trace()
                    pd.append(p_d)
                    pf.append(p_f)
                    g.append(_g)
                    auc.append(int(auroc))
                stats.append([src_name, int(np.mean(pd)), int(np.std(pd)),
                              int(np.mean(pf)), int(np.std(pf)),
                              int(np.mean(auc)), int(np.std(auc))])  # ,

        stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[-2], reverse=True),  # Sort by G Score
                                 columns=["Name", "Pd (Mean)", "Pd (Std)",
                                          "Pf (Mean)", "Pf (Std)",
                                          "AUC (Mean)", "AUC (Std)"])  # ,
        # "G (Mean)", "G (Std)"])
        print(tabulate(stats,
                       headers=["Name", "Pd (Mean)", "Pd (Std)",
                                "Pf (Mean)", "Pf (Std)",
                                "AUC (Mean)", "AUC (Std)"],
                       showindex="never",
                       tablefmt="fancy_grid"))

        result.update({tgt_name: stats})
    return result


def tnb_jur():
    all = get_all_datasets()
    for name, paths in all.iteritems():
        bellw(paths, paths, n_rep=10)
        set_trace()


if __name__ == "__main__":
    tnb_jur()
