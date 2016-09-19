from __future__ import print_function, division
import os
import sys

__root__ = os.path.join(os.getcwd().split('HDP')[0], 'HDP')
if __root__ not in sys.path:
    sys.path.append(__root__)


def pairs(s, t):
    """
    Takes 2 data sets A and B, returns all possible pairs of columns.
    :param a: First dataset, M columns.
    :param b: Second dataset, N columns.
    :return: a generator with pairs (m,n) (m,n -> M,N)
    """

    col_s, col_t = s.columns, t.columns
    for one in col_t:
        for two in col_s:
            yield two, one


class PrettyPrint(object):
    def __init__(self, header=[]):
        self.header = header
