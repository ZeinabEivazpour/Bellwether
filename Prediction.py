from __future__ import division

from pdb import set_trace

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from smote import SMOTE


def formatdata(tbl):
    """Change data structure to pandas"""
    try:
        Rows = [i.cells for i in tbl._rows]
        headers = [i.name for i in tbl.headers]
        return pd.DataFrame(Rows, columns=headers)
    except:
        Rows = [i.cells for i in tbl._rows]
        headers = [i.name for i in tbl.headers]
        return pd.DataFrame(Rows, columns=headers)


def Bugs(tbl):
    cells = [i.cells[-2] for i in tbl._rows]
    return cells


def rforest(train, test, tunings=None, smoteit=True, duplicate=True):
    """Random Forest"""
    # Apply random forest Classifier to predict the number of bugs.
    if smoteit:
        train = SMOTE(train, atleast=50, atmost=101, resample=duplicate)
    if not tunings:
        clf = RandomForestClassifier(n_estimators=100, random_state=1)
    else:
        clf = RandomForestClassifier(n_estimators=int(tunings[0]),
                                     max_features=tunings[1] / 100,
                                     min_samples_leaf=int(tunings[2]),
                                     min_samples_split=int(tunings[3]))
    traindf = formatdata(train)
    testdf = formatdata(test)
    columns = testdf.columns
    if len(traindf.columns) != len(traindf.columns):
        columns = traindf.columns if len(traindf.columns) < len(testdf.columns) else testdf.columns
    features = columns[:-2]
    klass = traindf[traindf.columns[-2]]
    clf.fit(traindf[features], klass)
    try:
        preds = clf.predict(testdf[features])
    except ValueError:
        set_trace()
    return preds


if __name__ == '__main__':
    pass
