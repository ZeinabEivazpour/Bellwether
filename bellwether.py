#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from texttable import Texttable

from Prediction import *
from logo import logo
from methods1 import *
from stats import ABCD


def getTunings(fname):
    raw = pd.read_csv('tunings.csv').transpose().values.tolist()
    formatd = pd.DataFrame(raw[1:], columns=raw[0])
    return formatd[fname].values.tolist()


class data:
    """Hold training and testing data"""

    def __init__(self, dataName='ant', type='jur'):
        if type == 'ant':
            dir = "./Data/Jureczko"
        elif type == 'jur':
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


class simulate:
    def __init__(self, file='ant', type='jur', tune=True):
        self.file = file
        self.type = type
        self.param = None if not tune else getTunings(file)
        # set_trace()

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
        # onlyMe = [self.file]
        val = []
        for file in train:

            try:
                fname = file[0].split('/')[-2]
            except:
                set_trace()

            header.append(fname)
            try:
                self.train = createTbl(file, isBin=True)
            except:
                set_trace()

            actual = Bugs(self.test)
            predicted = rforest(
                self.train,
                self.test,
                tunings=self.param,
                smoteit=True)
            p_buggy = [a for a in ABCD(before=actual, after=predicted).all()]
            val.append([fname, "%0.2f" % p_buggy[1].stats()[-2], "%0.2f" % p_buggy[1].stats()[-3]
                           , "%0.2f" % p_buggy[1].stats()[0], "%0.2f" % p_buggy[1].stats()[1]])

        table_rows.extend(sorted(val, key=lambda F: float(F[2]), reverse=True))

        "Pretty Print Thresholds"
        table = Texttable()
        table.set_cols_align(["l", 'l', "l", "l", "l"])
        table.set_cols_valign(["m", "m", "m", "m", "m"])
        table.set_cols_dtype(['t', 't', 't', 't', 't'])
        table.add_rows(table_rows)
        print(table.draw(), "\n")

        # ---------- DEBUG ----------
        #   set_trace()


def whatsInNasa():
    """Explore the Data set"""

    # for file in ["cm", "jm", "kc", "mc", "mw", "pc", "pc2"]:
    dir = './Data/mccabe'
    projects = [Name for _, Name, __ in walk(dir)][0]
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]
    try:
        ref = pd.read_csv(data[2][0]).columns.values.tolist()[1:]
    except:
        set_trace()
    for dat in data[1:]:
        for f in dat:
            raw = pd.read_csv(f)
            oldCol = raw.columns.values.tolist()
            newCol = [c for c in oldCol if c in ref]
            new = raw[newCol]
            new.to_csv(re.sub('mccabe', 'newMcCabes', f), index=False)
            # set_trace()


def nasa():
    for file in ["cm", "jm", "kc", "mc", "mw"]:
        print('### ' + file)
        simulate(file, type='nasa', tune=False).bellwether()


def jur():
    for file in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                 'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
        print('### ' + file)
        simulate(file, type='jur').bellwether()


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
    # jur()
    aeeem()
