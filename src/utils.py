from __future__ import print_function, division

import os
import sys
from random import shuffle

import pandas as pd

try:
    import cPickle as pickle
except ImportError:
    import pickle

import json
from StringIO import StringIO
import prettytable

__root__ = os.path.join(os.getcwd().split('HDP')[0], 'HDP')
if __root__ not in sys.path:
    sys.path.append(__root__)

"""
Some awesome utility functions used everywhere
"""


def stringify_pandas(pd):
    output = StringIO();
    pd.to_csv(output);
    output.seek(0);
    pt = prettytable.from_csv(output)
    return pt


def print_pandas(pd, op="text"):
    # if op.lower() is 'text':
    #     output = StringIO()
    #     pd.to_csv(output)
    #     output.seek(0)
    #     pt = prettytable.from_csv(output)
    #     print(pt)
    prefix = "\\begin{figure}\n\\centering\n\\resizebox{\\textwidth}{!}{"
    postfix = "}\n\\end{figure}"
    body = pd.to_latex()
    print(prefix+body+postfix)


def list2dataframe(lst):
    data = [pd.read_csv(elem) for elem in lst]
    return pd.concat(data, ignore_index=True)


def dump_json(result, dir='.', fname='data'):
    with open('{}/{}.json'.format(dir, fname), 'w+') as fp:
        json.dump(result, fp)


def load_json(path):
    with open(path) as data_file:
        data = json.load(data_file)

    return data


def brew_pickle(data, dir='.', fname='data'):
    with open('{}/{}.p'.format(dir, fname), 'w+') as fp:
        pickle.dump(data, fp)


def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def flatten(x):
    """
    Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
    and returns a single list [a,b,c,d,e,f]
    """
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def df_norm(dframe, type="min_max"):
    """ Normalize a dataframe"""
    if type == "min_max":
        return (dframe - dframe.min()) / (dframe.max() - dframe.min() + 1e-32)
    if type == "normal":
        return (dframe - dframe.mean()) / dframe.std()


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


def pretty(an_item):
    import pprint
    p = pprint.PrettyPrinter(indent=2)
    p.pprint(an_item)


def pairs(D):
    """
    Returns pairs of (key, values).
    :param D: Dictionary
    :return:
    """
    keys = D.keys()
    last = keys[0]
    for i in keys[1:]:
        yield D[last], D[i]
        last = i
