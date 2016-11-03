from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from prediction.model import rf_model
from old.stats import ABCD
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from numpy.random import choice
import numpy as np
from matching.bahsic.hsic import CHSIC
from sklearn.decomposition import TruncatedSVD as KernelSVD
from utils import *
from pdb import set_trace
from time import time


## Shut those god damn warnings up!
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_kernel_matrix(dframe, n_components=5):
    """
    This returns a Kernel Transformation Matrix $\Theta$ -> This is basically a mapping using PCA-SVD
    :param dframe: input data as a pandas dataframe.
    :param n_dim: Number of dimensions for the kernel matrix
    :return: $\Theta$ matrix
    """
    kernel = KernelSVD(n_components=5)
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
    S.columns = xrange(len(s_col))
    T = tgt[t_col]
    T.columns = xrange(len(t_col))

    all = pd.concat([S, T])
    N = int(min(len(S), len(T)))

    "Make sure x0, y0 are the same size. Note: This is rerun multiple times."
    if len(S) > len(T):
        x0, y0 = S.sample(n=N), T
    elif len(S) < len(T):
        x0, y0 = S, T.sample(n=N)
    else:
        x0, y0 = S, T

    mapper = get_kernel_matrix(all, n_components=5)

    x0 = pd.DataFrame(mapper.transform(x0), columns=xrange(n_components))
    y0 = pd.DataFrame(mapper.transform(y0), columns=xrange(n_components))

    x0.loc[:, src.columns[-1]] = pd.Series(src[src.columns[-1]], index=x0.index)
    y0.loc[:, tgt.columns[-1]] = pd.Series(src[src.columns[-1]], index=y0.index)

    return x0, y0


def cause_effect(x, y):
    """
    Run a non-parametric cause-effect test
    :param x: IID samples
    :param y: IID samples
    :return: A tuple (sign, delta-HSI). If X->Y: (+1, c_xy-c_yx) or if Y->X:(-1, ) else 0
    """

    def pack():
        """
        Split data into train test and pack them as tuples
        :return:
        """

        N = int(min(len(x), len(y)))
        "Make sure x0, y0 are the same size. Note: This is rerun multiple times."
        if len(x) > len(y):
            x0, y0 = choice(x, size=N), y
        elif len(x) < len(y):
            x0, y0 = x, choice(y, size=N)
        else:
            x0, y0 = x, y

        # Defaults to a 0.75/0.25 split.
        x_train, y_train, x_test, y_test = train_test_split(x0, y0)

        train = [(a, b) for a, b in zip(x_train, y_train)]
        test = [(a, b) for a, b in zip(x_test, y_test)]

        return train, test

    def unpack(lst, axis):
        return np.atleast_2d([l[axis] for l in lst]).T

    def residue(train, test, fwd=True):
        mdl = LinearRegression()
        if fwd:
            X = unpack(train, axis=0)
            y = unpack(train, axis=1)
            x_hat = unpack(test, axis=0)
            y_hat = unpack(test, axis=1)
        else:
            X = unpack(train, axis=1)
            y = unpack(train, axis=0)
            x_hat = unpack(test, axis=1)
            y_hat = unpack(test, axis=0)

        mdl.fit(X, y)
        return y_hat - mdl.predict(x_hat)

    train, test = pack()
    e_y = residue(train, test, fwd=True)
    e_x = residue(train, test, fwd=False)
    x_val = unpack(test, axis=0)
    y_val = unpack(test, axis=1)
    hsic = CHSIC()
    c_xy = hsic.UnBiasedHSIC(x_val, e_y)
    c_yx = hsic.UnBiasedHSIC(y_val, e_x)

    return abs(c_xy - c_yx) if c_xy < c_yx else 0


def metrics_match(src, tgt, n_redo):
    s_col = [col for col in src.columns[:-1] if not '?' in col]
    t_col = [col for col in tgt.columns[:-1] if not '?' in col]
    selected_col = dict()
    for t_metric in t_col:
        hi = -1e32
        for s_metric in s_col:
            value = np.median([cause_effect(src[s_metric], tgt[t_metric]) for _ in xrange(n_redo)])
            if value > hi:
                selected_col.update({t_metric: (s_metric, value)})
                hi = value

    return selected_col


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


def seer(source, target, n_rep=20, n_redo=5):
    """
    seer: Causal Inference Learning
    :param source:
    :param target:
    :return: result: A dictionary of estimated
    """
    result = dict()
    t0 = time()
    for tgt_name, tgt_path in target.iteritems():
        t1 = time()
        PD, PF, ED = [], [], []
        result.update({tgt_name: []})
        for src_name, src_path in source.iteritems():
            t2 = time()
            src = list2dataframe(src_path.data)
            tgt = list2dataframe(tgt_path.data)
            pd, pf, ed = [src_name], [src_name], [src_name]
            matched_src = metrics_match(src, tgt, n_redo)
            for n in xrange(n_rep):

                target_columns = []
                source_columns = []

                all_columns = [(key, val[0], val[1]) for key, val in matched_src.iteritems() if val[1] > 1]
                all_columns = sorted(all_columns, key=lambda x: x[-1])[::-1]  # Sort descending

                # Filter all columns to remove dupes
                for elem in all_columns:
                    if not elem[1] in source_columns:
                        target_columns.append(elem[0])
                        source_columns.append(elem[1])

                _train, __test = df_norm(src[source_columns + [src.columns[-1]]]), \
                                 df_norm(tgt[target_columns + [tgt.columns[-1]]])

                # _train, __test = map_transform(src[source_columns + [src.columns[-1]]],
                #                          tgt[target_columns + [tgt.columns[-1]]])

                actual, predicted = predict_defects(train=_train, test=__test)
                p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
                pd.append(p_buggy[1].stats()[0])
                pf.append(p_buggy[1].stats()[1])
                ed.append(p_buggy[1].stats()[-1])
            print("Time per source (s: {0}): {1:.2f}s".format(src_name, time()-t2))
            PD.append(pd)
            PF.append(pf)
            ED.append(ed)

        print("Time per target (t: {0}): {1:.2f}s".format(tgt_name, time()-t1))
        result[tgt_name].append((PD, PF))
    
    print("Time per call: {0:.2f}s".format(time()-t0))
    return result
