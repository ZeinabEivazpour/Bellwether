import os
import sys

root = os.path.join(os.getcwd().split('smells')[0], 'smells')
if root not in sys.path:
    sys.path.append(root)
from utils import explore
from pdb import set_trace
from glob import glob


class _Data:
    """Hold training and testing data"""

    def __init__(self, dataName, type):
        if type == 'DataClass':
            directory = os.path.join(root, "data/DataClass")
        elif type == 'FeatureEnvy':
            directory = os.path.join(root, "data/FeatureEnvy")
        elif type == 'GodClass':
            directory = os.path.join(root, "data/GodClass")
        elif type == "LongMethod":
            directory = os.path.join(root, "data/LongMethod")

        files =  glob(os.path.join(os.path.abspath(directory), "*.csv"))



class DataClass:
    "NASA"
    def __init__(self):
        self.projects = {}
        for file in ["cm", "jm", "kc", "mc", "mw"]:
            self.projects.update({file: _Data(dataName=file, type='DataClass')})


class FeatureEnvy:
    "Apache"
    def __init__(self):
        self.projects = {}
        for file in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                     'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
             self.projects.update({file: _Data(dataName=file, type='FeatureEnvy')})


class GodClass:
    "AEEEM"
    def __init__(self):
        self.projects = {}
        for file in ["EQ", "JDT", "LC", "ML", "PDE"]:
            self.projects.update({file: _Data(dataName=file, type='GodClass')})


class LongMethod:
    "RELINK"
    def __init__(self):
        self.projects = {}
        for file in ["Apache", "Safe", "Zxing"]:
            self.projects.update({file: _Data(dataName=file, type='LongMethod')})

def get_all_projects():
    all = dict()
    for community in [DataClass, FeatureEnvy, GodClass, LongMethod]:
        all.update({community.__doc__: community().projects})
    return all

def _test():
    data = FeatureEnvy()
    data.projects


if __name__ == "__main__":
    _test()
