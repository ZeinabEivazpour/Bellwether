from __future__ import print_function, division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from utils import pretty
from old.Prediction import rforest
from data.handler import get_all_projects
from matching.match_metrics import match_metrics, list2dataframe
from pdb import set_trace
from prediction.model import rf_model, logistic_model
from old.methods1 import createTbl
from old.stats import ABCD
import pickle


class HDP:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    @staticmethod
    def known_bellwether(datasets):
        """
        Returns the predetermined bellwether for the community
        """
        for key, value in datasets.iteritems():
            if key.lower() in ['lc', 'mc', 'lucene', 'safe']:
                return key, value

    def matching(self, bellwether=False):
        # source_bw = self.bellwether(self.source)
        all_matches = {
            "Source": {
                "name": [],
                "path": []
            },
            "Target": {
                "name": [],
                "path": []
            }
            # ,
            # "Matches": []
        }
        bw_matches = []

        for tgt_name, tgt_path in self.target.iteritems():
            src_name, src_path = self.known_bellwether(self.source)
            matched = match_metrics(src_path, tgt_path)
            if matched:
                bw_matches.extend(matched)
        bw_matches = list(set(bw_matches))

        # print(len(bw_matches))

        for tgt_name, tgt_path in self.target.iteritems():
            for src_name, src_path in self.source.iteritems():
                matched = match_metrics(src_path, tgt_path)
                if matched:
                    # print("S: {} | T: {} | {}".format(src_name, tgt_name, len(matched)))
                    all_matches["Source"]["name"].append(src_name)
                    all_matches["Source"]["path"].append(src_path)
                    all_matches["Target"]["name"].append(tgt_name)
                    all_matches["Target"]["path"].append(tgt_path)

        return all_matches, bw_matches

    def process(self):
        data, bw_matches = self.matching(bellwether=True)
        source, target = data["Source"], data["Target"]

        for s_name, s_path, t_name, t_path in zip(source["name"], source["path"],
                                                  target["name"], target["path"]):
            print("S: {} | T: {}".format(s_name, t_name), end="")
            train = list2dataframe(s_path.data)
            test = list2dataframe(t_path.data)
            train_klass = train.columns[-1]
            test_klass = test.columns[-1]
            trainCol, testCol = [], []

            for col in bw_matches:
                if not col[0] in trainCol and not col[1] in testCol:
                    trainCol.append(col[0])
                    testCol.append(col[1])

            # for col in match:
            # trainCol.append(col[0])
            # testCol.append(col[1])
            # set_trace()

            train = train[trainCol + [train_klass]]
            test = test[testCol + [test_klass]]
            actual = test[test.columns[-1]].values.tolist()
            predicted = logistic_model(train, test)
            p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
            print("| Val: ", p_buggy[1].stats()[-1])
            yield t_name, p_buggy[1].stats()[-1]


def save(obj):
    pickle.dump(obj, open("./picklejar/result_dump.pkl", "wb"))


def load(name):
    return pickle.load(open("./picklejar/{}.pkl".format(name), "rb"))


def run_hdp():
    all = get_all_projects()
    result = {}
    for k, v in all.iteritems():
        for kk in v.keys():
            result.update({kk: []})

    for key_s, value_s in all.iteritems():
        for key_t, value_t in all.iteritems():
            if not key_s == key_t:
                print("Source: {}, Target: {}".format(key_s, key_t))
                # HDP(value_s, value_t).process()
                for name, auc in HDP(value_s, value_t).process():
                    result[name].append(auc)
    save(result)


def get_stats():
    import pickle
    import numpy as np
    result = pickle.load(open('./picklejar/result.pkl', 'rb'))
    for k, v in result.iteritems():
        auc = np.mean(v)
        stdev = np.var(v) ** 0.5
        print("{},{},{}".format(k, round(auc, 2), round(stdev, 2)))


if __name__ == "__main__":
    # get_stats()
    run_hdp()
