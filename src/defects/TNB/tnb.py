from __future__ import print_function, division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
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


def tnb():
    """
    TCA: Transfer Component Analysis
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
                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)
                pd, pf, g, auc = [], [], [], []
                dcv_src, dcv_tgt = get_dcv(src, tgt)

                for _ in xrange(n_rep):
                    recall, loc = None, None
                    norm_src, norm_tgt = smart_norm(src, tgt, dcv_src, dcv_tgt)
                    _train, __test = map_transform(norm_src, norm_tgt)
                    # for k in np.arange(0.1,1,0.1):
                    actual, predicted, distribution = predict_defects(train=_train, test=__test)
                    loc = tgt["$loc"].values
                    loc = loc * 100 / np.max(loc)
                    recall, loc, au_roc = get_curve(loc, actual, predicted)
                    effort_plot(recall, loc,
                                save_dest=os.path.abspath(os.path.join(root, "plot", "plots", tgt_name)),
                                save_name=src_name)
                    p_d, p_f, p_r, rc, f_1, e_d, _g, _ = abcd(actual, predicted, distribution)

                    pd.append(p_d)
                    pf.append(p_f)
                    g.append(_g)
                    auc.append(int(au_roc))

                    # set_trace()

                stats.append([src_name, int(np.mean(pd)), int(np.std(pd)),
                              int(np.mean(pf)), int(np.std(pf)),
                              int(np.mean(auc)), int(np.std(auc))])  # ,
                # int(np.mean(g)), int(np.std(g))])

        stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[0]),  # Sort by G Score
                                 columns=["Name", "Pd (Mean)", "Pd (Std)",
                                          "Pf (Mean)", "Pf (Std)",
                                          "AUC (Mean)", "AUC (Std)"])  # ,
        # "G (Mean)", "G (Std)"])
        result.update({tgt_name: stats})
    # set_trace()
    return result


def tnb_jur():
    from data.handler import get_all_projects
    all = get_all_projects()
    apache = all["Apache"]
    return tnb(apache, apache, n_rep=1)
    pass


if __name__ == "__main__":
    tnb_jur()
