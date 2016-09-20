from __future__ import division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from old.Prediction import rforest
from data.handler import *
from utils import formatData


def rf_model(source, target):
    for _ in xrange(10):
        predicted = rforest(train, test, tunings=None)
    pass


def _test_model():

    pass


if __name__ == '__main__':
    _test_model()
