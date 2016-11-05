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
from stats.scott_knott import sk_chart
from metrics.abcd import abcd
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import *
from mklaren.projection.nystrom import Nystrom

## Shut those god damn warnings up!
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_kernel_matrix(dframe, n_dim=15):
    """
    This returns a Kernel Transformation Matrix $\Theta$

    It uses kernel approximation offered by the MKlaren package
    For the sake of completeness (and for my peace of mind, I use the best possible approx.)

    :param dframe: input data as a pandas dataframe.
    :param n_dim: Number of dimensions for the kernel matrix (default=15)
    :return: $\Theta$ matrix
    """
    ker = Kinterface(data=dframe.values, kernel=rbf_kernel, kernel_args={"sigma": 30})
    model = Nystrom(rank=n_dim)
    try:
        model.fit(ker)
    except:
        set_trace()
    g_nystrom = model.G
    # err = np.linalg.norm(ker[:, :] - g_nystrom.dot(g_nystrom.T))
    return g_nystrom


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
    N = int(min(len(S), len(T)))

    "Make sure x0, y0 are the same size. Note: This is rerun multiple times."
    if len(S) > len(T):
        x0, y0 = S.sample(n=N), T
    elif len(S) < len(T):
        x0, y0 = S, T.sample(n=N)
    else:
        x0, y0 = S, T

    x0 = pd.DataFrame(get_kernel_matrix(x0, n_components), columns=xrange(n_components))
    y0 = pd.DataFrame(get_kernel_matrix(y0, n_components), columns=xrange(n_components))

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
    try:  ## !!GUARD: PLEASE REMOVE AFTER DEBUGGING!!
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


def tca_plus(source, target, n_rep=1):
    """
    TCA+: Transfer Component Analysis
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    result = dict()

    for tgt_name, tgt_path in target.iteritems():
        PD, PF, F1, G = [], [], [], []
        result.update({tgt_name: []})
        print("Target Project: {}\n".format(tgt_name), end="```\n")
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:
                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)
                pd, pf, f1, g = [src_name], [src_name], [src_name], [src_name]
                for _ in xrange(n_rep):
                    dcv_src, dcv_tgt = get_dcv(src, tgt)
                    norm_src, norm_tgt = smart_norm(src, tgt, dcv_src, dcv_tgt)
                    _train, __test = map_transform(norm_src, norm_tgt)
                    actual, predicted = predict_defects(train=_train, test=__test)
                    p_d, p_f, p_r, rc, f_1, e_d, _g = abcd(actual, predicted)
                    pd.append(p_d)
                    pf.append(p_f)
                    f1.append(f_1)
                    g.append(_g)

                PD.append(pd)
                PF.append(pf)
                F1.append(f1)
                G.append(g)

        result[tgt_name].append((PD, PF, F1, G))

    return result
