from __future__ import division
import os
import sys
import pandas as pd

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from old.Prediction import rforest
from data.handler import *


def getTunings(fname):
    raw = pd.read_csv(root+'/old/tunings.csv').transpose().values.tolist()
    formatd = pd.DataFrame(raw[1:], columns=raw[0])
    return formatd[fname].values.tolist()


def df2thing(dframe):
    from glob import glob
    if len(glob('temp.csv')): os.remove('temp.csv')
    dframe.to_csv('temp.csv', index=False)
    return createTbl(['temp.csv'], isBin=True)


def rf_model(source, target, name):
    train = df2thing(source)
    test = df2thing(target)
    return rforest(train, test, tunings=getTunings(name))


#
# def logistic_model(source, target):
#     train = df2thing(source)
#     test = df2thing(target)
#     return logistic_regression(train, test)


def _test_model():
    pass


if __name__ == '__main__':
    _test_model()
