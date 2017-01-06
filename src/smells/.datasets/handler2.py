import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src/smells')
if root not in sys.path:
    sys.path.append(root)
from utils import explore
from pdb import set_trace
from glob import glob


class _Data:
    """Hold training and testing data"""

    def __init__(self, dataName, type):
        if type == 'class':
            directory = os.path.join(root, "data/class")
        elif type == 'method':
            directory = os.path.join(root, "data/method")

        # if type == 'DataClass':
        #     directory = os.path.join(root, "data/DataClass")
        # elif type == 'FeatureEnvy':
        #     directory = os.path.join(root, "data/FeatureEnvy")
        # elif type == 'GodClass':
        #     directory = os.path.join(root, "data/GodClass")
        # elif type == "LongMethod":
        #     directory = os.path.join(root, "data/LongMethod")

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

def get_all_datasets():
    dir = os.path.abspath(os.path.join(root, "datasets"))
    datasets = dict()
    for datapath in  os.listdir(dir):
        formatted_path = os.path.join(dir, datapath)
        if os.path.isdir(formatted_path):
            datasets.update({datapath: dict()})
            files = glob(os.path.join(formatted_path, "*.csv"))
            for f in files:
                fname = f.split('/')[-1].split("-")[0]
                datasets[datapath].update({fname: f})

    return datasets


def _test():
    data = FeatureEnvy()
    data.projects


if __name__ == "__main__":
    _test()
