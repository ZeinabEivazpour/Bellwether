#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

import os
import sys

from texttable import Texttable

from Prediction import *
from logo import logo
from methods1 import *
from stats import ABCD
import pandas
from py_weka.classifier import classify
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def getTunings(fname):
    raw = pd.read_csv(root + '/old/tunings.csv').transpose().values.tolist()
    formatd = pd.DataFrame(raw[1:], columns=raw[0])
    return formatd[fname].values.tolist()


class data:
    """Hold training and testing data"""

    def __init__(self, dataName='ant', type='jur'):
        if type == 'jur':
            dir = "./Data/Jureczko"
        elif type == 'nasa':
            dir = "./Data/mccabe"
        elif type == 'aeeem':
            dir = "./Data/AEEEM"
        elif type == "relink":
            dir = './Data/Relink'
        elif type == 'other':
            dir = './Data/other/'
        try:
            projects = [Name for _, Name, __ in walk(dir)][0]
        except:
            set_trace()
        numData = len(projects)  # Number of data
        one, two = explore(dir)
        data = [one[i] + two[i] for i in xrange(len(one))]

        def whereis():
            for indx, name in enumerate(projects):
                if name == dataName:
                    return indx

        loc = whereis()
        self.train = []
        self.test = []
        for idx, dat in enumerate(data):
            if idx != loc:
                self.train.append(dat)
        self.test = data[loc]


class simulate:
    def __init__(self, file='ant', type='jur', tune=True):
        self.file = file
        self.type = type
        self.param = None if not tune else getTunings(file)

    def incrUpdates(self):
        """When to update a reference set"""
        src = data(dataName=self.file, type=self.type)
        self.test = createTbl(src.test, isBin=True)

    def bellwether(self):
        src = data(dataName=self.file, type=self.type)
        self.test = createTbl(src.test, isBin=True)
        if len(src.train) == 1:
            train = src.train[0]
        else:
            train = src.train
        header = [" "]
        stats = []
        for file in train:
            fname = file[0].split('/')[-2]
            e = []
            header.append(fname)
            self.train = createTbl(file, isBin=True)
            actual = Bugs(self.test)
            pd, pf, e_d, val = [], [], [], []
            for _ in xrange(1):
                predicted = classify(
                    self.train,
                    self.test)#,
                    # tunings=self.param)
                p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
                e.append([p_buggy[1].stats()[-1]])

                pd.append(p_buggy[1].stats()[0])
                pf.append(p_buggy[1].stats()[1])
                e_d.append(p_buggy[1].stats()[-2])
            stats.append([fname, int(np.mean(pd)), int(np.std(pd)),
                          int(np.mean(pf)), int(np.std(pf)),
                          int(np.mean(e_d)), round(np.std(e_d))])

        stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[-2], reverse=True),
                                 columns=["Name", "Pd (Mean)", "Pd (Std)", "Pf (Mean)", "Pf (Std)", "G (Mean)",
                                          "G (Std)"])

        return stats


def attributes(type='jur'):
    """ Returns total instances, and no. of bug in a file"""
    from Prediction import formatdata
    if type == 'jur':
        for name in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                     'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
            src = data(dataName=name, type=type)
            test = src.test
            dat = createTbl(test, isBin=True)
            dframe = formatdata(dat)
            bugs = dframe[dframe.columns[-2]].values
            print(name, len(bugs), str(sum(bugs)) + " (%0.2f)" % (sum(bugs) / len(bugs) * 100))
    elif type == 'nasa':
        for name in ["cm", "jm", "kc", "mc", "mw"]:
            src = data(dataName=name, type=type)
            test = src.test
            dat = createTbl(test, isBin=True)
            dframe = formatdata(dat)
            bugs = dframe[dframe.columns[-2]].values
            print(name, len(bugs), str(sum(bugs)) + " (%0.2f)" % (sum(bugs) / len(bugs) * 100))
    elif type == 'aeeem':
        for name in ["EQ", "JDT", "LC", "ML", "PDE"]:
            src = data(dataName=name, type=type)
            test = src.test
            dat = createTbl(test, isBin=True)
            dframe = formatdata(dat)
            bugs = dframe[dframe.columns[-2]].values
            print(name, len(bugs), str(sum(bugs)) + " (%0.2f)" % (sum(bugs) / len(bugs) * 100))
    elif type == 'relink':
        for name in ["Apache", "Safe", "Zxing"]:
            src = data(dataName=name, type=type)
            test = src.test
            dat = createTbl(test, isBin=True)
            dframe = formatdata(dat)
            bugs = dframe[dframe.columns[-2]].values
            print(name, len(bugs), str(sum(bugs)) + " (%0.2f)" % (sum(bugs) / len(bugs) * 100))


def nasa():
    print("NASA\n------\n```")
    for file in ["cm", "jm", "kc", "mc", "mw"]:
        print('### ' + file)
        simulate(file, type='nasa', tune=False).bellwether()
    print('```')


def jur():
    result = {}
    for file in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                 'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
        result.update({file: simulate(file, type='jur').bellwether()})

    return result


def aeeem():
    print("AEEEM\n------\n```")
    for file in ["EQ", "JDT", "LC", "ML", "PDE"]:
        print('### ' + file)
        simulate(file, type='aeeem', tune=False).bellwether()
    print('```')


def relink():
    print("Relink\n------\n```")
    for file in ["Apache", "Safe", "Zxing"]:
        print('### ' + file)
        simulate(file, type='relink', tune=False).bellwether()
    print("```")


def other():
    print("Other\n------\n```")
    files = [x[0].split('/')[-1] for x in walk('Data/other/')][1:]
    for file in files:
        print('### ' + file)
        simulate(file, type='other', tune=False).bellwether()
    print("```")


if __name__ == "__main__":
    logo()  # Print logo
    # other()
    # nasa()
    jur()
    # aeeem()
    # relink()
    # attributes('jur')
    # attributes('nasa')
    # attributes('aeeem')
    # print("")
    # attributes('relink')
