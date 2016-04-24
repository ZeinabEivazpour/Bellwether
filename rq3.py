#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from Prediction import *
from logo import logo
from methods1 import *
from sk import rdivDemo
from stats import ABCD


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

def knn(train, test):
    lot = train._rows
    new = []

    def edist(A, B):
        return np.sqrt(sum([a**2-b**2 for a, b in zip(A.cells[:-2], B.cells[:-2])]))

    for samp in test._rows:
        new.extend(sorted(lot, key=lambda X: edist(X, samp))[:10])

    new = list(set(new))
    return new

class makeDataSets:
    def __init__(self, file='ant', type='jur', tune=True):
        self.file = file
        self.type = type

    def barak09(self):
        src = data(dataName=self.file, type=self.type)
        self.test = createTbl(src.test, isBin=True)

        def flat(lst):
            new = []
            for l in lst:
                new.extend(l)
            return new

        self.train = createTbl(flat(src.train), isBin=True)
        new = knn(self.train, self.test)

        # Save the KNN data as text file
        Rows = [i.cells[:-1] for i in new]
        headers = [i.name for i in self.train.headers]
        try:
            File = pd.DataFrame(Rows, columns=headers[:-1])
        except:
            set_trace()

        try:
            File.to_csv(self.file + '.csv', index=False)
        except:
            set_trace()


class simulate:
    def __init__(self, file='ant', type='jur', tune=True):
        self.file = file
        self.type = type

    def turhan09(self):
        pass

    def turhan11(self):
        pass

    def bellwether(self):
        everything = []
        src = data(dataName=self.file, type=self.type)
        self.test = createTbl(src.test, isBin=True)
        table_rows = [["Dataset", "ED", "G2", "Pd", "Pf"]]

        if len(src.train) == 1:
            train = src.train[0]
        else:
            train = src.train

        header = [" "]
        val = []
        for file in train:
            fname = file[0].split('/')[-2]
            e = [fname]
            header.append(fname)
            self.train = createTbl(file, isBin=True)
            actual = Bugs(self.test)
            for _ in xrange(10):
                predicted = rforest(
                    self.train,
                    self.test,
                    tunings=self.param,
                    smoteit=True)
                p_buggy = [a for a in ABCD(before=actual, after=predicted).all()]
                e.append(p_buggy[1].stats()[-2])
            everything.append(e)

        rdivDemo(everything, isLatex=True)


def nasa():
    print("NASA\n------\n```")
    for file in ["cm", "jm", "kc", "mc", "mw"]:
        print('### ' + file)
        simulate(file, type='nasa', tune=False).bellwether()
    print('```')


def jur():
    print("Jureczko\n------\n```")
    for file in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                 'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
        print('### ' + file)
        simulate(file, type='jur').barakFilter()
    print('```')


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


if __name__ == "__main__":
    logo()  # Print logo
    # nasa()
    jur()
    # aeeem()
    # relink()
    # attributes('jur')
    # attributes('nasa')
    # attributes('aeeem')
    # print("")
    # attributes('relink')
