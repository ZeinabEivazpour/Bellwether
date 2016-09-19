import os
import sys

root = os.path.join(os.getcwd().split('HDP')[0], 'HDP')
if root not in sys.path:
    sys.path.append(root)
from pdb import set_trace

import pandas as pd

class dataset(object):
    def __init__(self):
        pass
    def getRandomDataSet():
        pass

class AEEEM(dataset):
    def __init__(self):
        self.dirpath = './AEEEM/'
        self.projects = 
        pass

class Jureczko(dataset):

class ReLink(dataset):

class NASA(dataset):

if __name__ == "__main__":
