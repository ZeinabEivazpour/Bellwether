from __future__ import division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from old.Prediction import rforest
from data.handler import *
from pdb import set_trace


def run_hdp():
    for community in [AEEEM, Jureczko, AEEEM, ReLink, NASA]:
        proj = community().projects
        set_trace()


if __name__ == "__main__":
    run_hdp()
