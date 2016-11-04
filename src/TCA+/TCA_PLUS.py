from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from data.handler import get_all_projects
from prediction.model import rf_model
from pdb import set_trace
from utils import *
from scipy.spatial.distance import euclidean
from sklearn.decomposition import TruncatedSVD as KernelSVD
from numpy.random import choice
import numpy as np
from old.sk import rdivDemo
from old.stats import ABCD

def get_kernel_matrix(dframe, n_components=5):
    """
    This returns a Kernel Transformation Matrix $\Theta$ -> This is basically a mapping using PCA-SVD
    :param dframe: input data as a pandas dataframe.
    :param n_dim: Number of dimensions for the kernel matrix
    :return: $\Theta$ matrix
    """
    kernel = KernelSVD(n_components)
    kernel.fit(dframe)
    return kernel


def map_transform(src, tgt, n_components=5):
    """
    Run a map and transform x and y onto a new space using TCA
    :param src: IID samples
    :param tgt: IID samples
    :return: Mapped x and y
    """
    s_col = [col for col in src.columns[:-1] if '?' not in col]
    t_col = [col for col in tgt.columns[:-1] if '?' not in col]
    S = src[s_col]
    T = tgt[t_col]
    all = pd.concat([S,T])

    N = int(min(len(S), len(T)))

    "Make sure x0, y0 are the same size. Note: This is rerun multiple times."
    if len(S) > len(T):
        x0, y0 = S.sample(n=N), T
    elif len(S) < len(T):
        x0, y0 = S, T.sample(n=N)
    else:
        x0, y0 = S, T

    mapper = get_kernel_matrix(all, n_components)

    x0 = pd.DataFrame(mapper.transform(x0), columns=xrange(n_components))
    y0 = pd.DataFrame(mapper.transform(y0), columns=xrange(n_components))

    x0.loc[:, src.columns[-1]] = pd.Series(src[src.columns[-1]], index=x0.index)
    y0.loc[:, tgt.columns[-1]] = pd.Series(src[src.columns[-1]], index=y0.index)

    return x0, y0


def predict_defects(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    # Binarize data
    train.loc[train[train.columns[-1]] > 0, train.columns[-1]] = 1
    test.loc[test[test.columns[-1]] > 0, test.columns[-1]] = 1

    actual = test[test.columns[-1]].values.tolist()

    predicted = rf_model(train, test)
    return actual, predicted


def get_dcv(src, tgt):
    """Get dataset characteristic vector."""
    s_col = [col for col in src.columns[:-1] if '?' not in col]
    t_col = [col for col in tgt.columns[:-1] if '?' not in col]
    S = src[s_col]
    T = tgt[t_col]

    def self_dist_mtx(arr):
        dist = []
        for a, a_val in enumerate(arr[:-1]):
            for b, b_val in enumerate(arr[1:]):
                if a < b:
                    dist.append(euclidean(a, b))
        return dist

    dist_src = self_dist_mtx(S.values)
    dist_tgt = self_dist_mtx(T.values)

    dcv_src = [np.mean(dist_src), np.median(dist_src), np.min(dist_src), np.max(dist_src), np.std(dist_src),
               len(S.values)]
    dcv_tgt = [np.mean(dist_tgt), np.median(dist_tgt), np.min(dist_tgt), np.max(dist_tgt), np.std(dist_tgt),
               len(T.values)]
    return dcv_src, dcv_tgt


def sim(c_s, c_t, e=0):
    if c_s[e] * 1.6 < c_t[e]:
        return "VH"  # Very High
    if c_s[e] * 1.3 < c_t[e] <= c_s[e] * 1.6:
        return "H"  # High
    if c_s[e] * 1.1 < c_t[e] <= c_s[e] * 1.3:
        return "SH"  # Slightly High
    if c_s[e] * 0.9 <= c_t[e] <= c_s[e] * 1.1:
        return "S"  # Same
    if c_s[e] * 0.7 <= c_t[e] < c_s[e] * 0.9:
        return "SL"  # Slightly Low
    if c_s[e] * 0.4 <= c_t[e] < c_s[e] * 0.7:
        return "L"  # Low
    if c_t[e] < c_s[e] * 0.4:
        return "VL"  # Very Low


def smart_norm(src, tgt, c_s, c_t):
    """
    ARE THESE NORMS CORRECT?? OPEN AN ISSUE REPORT TO VERIFY
    :param src:
    :param tgt:
    :param c_s:
    :param c_t:
    :return:
    """
    try: ## !!GUARD: PLEASE REMOVE AFTER DEBUGGING!!
        # Rule 1
        if sim(c_s, c_t, e=0) == "S" and sim(c_s, c_t, e=-2) == "S":
            return src, tgt

        # Rule 2
        elif sim(c_s, c_t, e=2) == "VL" or "VH" \
                and sim(c_s, c_t, e=3) == "VL" or "VH" \
                and sim(c_s, c_t, e=-1) == "VL" or "VH":
            return df_norm(src), df_norm(tgt)

        # Rule 3.1
        elif sim(c_s, c_t, e=-2) == "VH" and c_s[-1] > c_t[-1] or \
                                sim(c_s, c_t, e=-2) == "VL" and c_s[-1] < c_t[-1]:
            return df_norm(src, type="normal"), df_norm(tgt)

        # Rule 4
        elif sim(c_s, c_t, e=-2) == "VH" and c_s[-1] < c_t[-1] or \
                                sim(c_s, c_t, e=-2) == "VL" and c_s[-1] > c_t[-1]:
            return df_norm(src), df_norm(tgt, type="normal")
        else:
            return df_norm(src, type="normal"), df_norm(tgt, type="normal")
    except:
        return src, tgt


def tca_plus(source, target):
    """
    TCA+: Transfer Component Analysis
    :param source:
    :param target:
    :return:
    """
    for tgt_name, tgt_path in target.iteritems():
        PD, PF, ED = [], [], []
        print("Target Project: {}\n".format(tgt_name), end="```\n")
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:
                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)
                pd, pf, ed = [src_name], [src_name], [src_name]
                for _ in xrange(3):
                    dcv_src, dcv_tgt = get_dcv(src, tgt)
                    norm_src, norm_tgt = smart_norm(src, tgt, dcv_src, dcv_tgt)
                    _train, __test = map_transform(norm_src, norm_tgt)
                    actual, predicted = predict_defects(train=_train, test=__test)
                    p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
                    pd.append(p_buggy[1].stats()[0])
                    pf.append(p_buggy[1].stats()[1])
                    ed.append(p_buggy[1].stats()[-1])

                PD.append(pd)
                PF.append(pf)
                ED.append(ed)

        rdivDemo(ED, isLatex=False)
        # set_trace()
        print('```')


def execute():
    """
    This method performs HDP.
    :return:
    """
    all_projects = get_all_projects()  # Get a dictionary of all projects and their respective pathnames.
    result = {}  # Store results here

    for target in all_projects.keys():
        for source in all_projects.keys():
            if source == target:  # This ensures transfer happens within community
                print("Target Community: {} | Source Community: {}".format(target, source))
                tca_plus(all_projects[source], all_projects[target])


def __test_tca():
    """
    A test case goes here.
    :return:
    """


if __name__ == "__main__":
    execute()
