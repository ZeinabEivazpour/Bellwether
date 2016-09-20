from __future__ import print_function, division
import os
import sys
from random import shuffle
import pandas as pd

__root__ = os.path.join(os.getcwd().split('HDP')[0], 'HDP')
if __root__ not in sys.path:
    sys.path.append(__root__)


def explore(dir):
    datasets = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        datasets.append(dirpath)

    dataset = []
    for k in datasets[1:]:
        data0 = [[dirPath, fname] for dirPath, _, fname in os.walk(k)]
        dataset.append(
            [data0[0][0] + '/' + p for p in data0[0][1] if not p == '.DS_Store'])
    return dataset


def formatData(tbl, picksome=False, addsome=None):
    """ Convert Tbl to Pandas DataFrame

    :param tbl: Thing object created using function createTbl
    :returns table in a DataFrame format
    """
    some = []
    Rows = [i.cells for i in tbl._rows]
    if picksome:
        shuffle(Rows)
        for i in xrange(int(len(Rows) * 0.1)):
            some.append(Rows.pop())
    headers = [i.name for i in tbl.headers]

    if addsome:
        Rows.extend(addsome)

    if picksome:
        return pd.DataFrame(Rows, columns=headers), some
    else:
        return pd.DataFrame(Rows, columns=headers)


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
