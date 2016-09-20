from __future__ import division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from old.Prediction import rforest, logistic_regression
from data.handler import *
from utils import formatData


def df2thing(dframe):
    dframe.to_csv('temp.csv', index=False)
    return createTbl(['temp.csv'], isBin=True)


def rf_model(source, target):
    train = df2thing(source)
    test = df2thing(target)
    return rforest(train, test, tunings=None)


def logistic_model(source, target):
    train = df2thing(source)
    test = df2thing(target)
    return logistic_regression(train, test)


def _test_model():
    pass


if __name__ == '__main__':
    _test_model()
