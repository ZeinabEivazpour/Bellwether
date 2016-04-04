from __future__ import division

import pandas as pd
from abcd import _Abcd
from sklearn.ensemble import RandomForestClassifier

from methods1 import *
from smote import *


def formatData(tbl):
    Rows = [i.cells for i in tbl._rows]
    headers = [i.name for i in tbl.headers]
    return pd.DataFrame(Rows, columns=headers)


def Bugs(tbl):
    cells = [i.cells[-2] for i in tbl._rows]
    return cells


def rforest(train, test, tunings=None, smoteit=True, duplicate=True):
    "RF "
    # Apply random forest Classifier to predict the number of bugs.
    # train = createTbl(train)
    # test = createTbl(test)

    if smoteit:
        train = SMOTE(train, atleast=50, atmost=101, resample=duplicate)
    if not tunings:
        clf = RandomForestClassifier(n_estimators=100, random_state=1)
    else:
        clf = RandomForestClassifier(n_estimators=int(tunings[0]),
                                     max_features=tunings[1] / 100,
                                     min_samples_leaf=int(tunings[2]),
                                     min_samples_split=int(tunings[3])
                                     )
    train_DF = formatData(train)
    test_DF = formatData(test)
    features = train_DF.columns[:-2]
    klass = train_DF[train_DF.columns[-2]]
    # set_trace()
    clf.fit(train_DF[features], klass)
    preds = clf.predict(test_DF[test_DF.columns[:-2]])
    return preds


def _RF():
    "Test RF"
    dir = '../Data'
    one, two = explore(dir)
    # Training data
    train_DF = createTbl([one[0][0]])
    # Test data
    test_df = createTbl([one[0][1]])
    actual = Bugs(test_df)
    preds = rforest(train_DF, test_df, mss=6, msl=8,
                    max_feat=4, n_est=5756,
                    smoteit=False)
    print _Abcd(before=actual, after=preds, show=False)[-1]


if __name__ == '__main__':
    random.seed(0)
    Dat = []
    for _ in xrange(10):
        print(_where2pred())
# Dat.insert(0, 'Where2 untuned')
#  rdivDemo([Dat])
