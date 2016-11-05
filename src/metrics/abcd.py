from __future__ import division, print_function
from sklearn.metrics import *
from pdb import set_trace
import numpy as np


def abcd(actual, predicted):
    """
    Confusion Matrix:

    |`````````````|`````````````|
    |  TP[0][0]   |  FN[0][1]   |
    |             |             |
    |`````````````|`````````````|
    |  FP[1][0]   |  TN[1][1]   |
    |             |             |
    `````````````````````````````
    """

    def stringify(lst): return [str(a) for a in lst]

    actual = stringify(actual)
    predicted = stringify(predicted)
    c_mtx = confusion_matrix(actual, predicted)

    Pd = c_mtx[0][0] / (c_mtx[0][0] + c_mtx[0][1])  # TP/(TP+FN)
    Pf = c_mtx[1][0] / (c_mtx[1][0] + c_mtx[1][1])  # FP/(FP+TN)
    Pr = c_mtx[0][0] / (c_mtx[0][0] + c_mtx[1][0])  # FP/(TP+FP)
    Rc = c_mtx[0][0] / (c_mtx[0][0] + c_mtx[0][1])  # TP/(TP+FN)
    F1 = 2 * c_mtx[0][0] / (2 * c_mtx[0][0] + c_mtx[1][0] + c_mtx[0][1])  # F1 = 2*TP/(2*TP+FP+FN)
    Ed = np.sqrt(0.7 * (1 - Pd) ** 2 + 0.3 * Pf ** 2)
    G = np.sqrt(Pd*Pf) # Harmonic Mean between Pd, Pf
    return Pd, Pf, Pr, Rc, F1, Ed, G


def print_stats(actual, predicted):
    print("PD  ", "PF  ", "Prec", "Rec ", "F1  ", "Bal ", " G  ", sep="\t")
    print("{0:0.2f}\t{1:0.2f}\t{2:0.2f}\t{3:0.2f}\t{4:0.2f}\t{5:0.2f}\t{6:0.2f}".format(*abcd(actual, predicted)), )


def _test_abcd():
    x = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]
    stats = abcd(x, y)
    print_stats(x, y)


if __name__ == "__main__":
    _test_abcd()
