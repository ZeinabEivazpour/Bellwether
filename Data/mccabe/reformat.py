"""
Change the dependent variable column to 1's and 0's
"""
import pandas as pd
import numpy  as np
from os import walk
from pdb import set_trace


def explore(dir):
  datasets = []
  for (dirpath, dirnames, filenames) in walk(dir):
    datasets.append(dirpath)

  training = []
  testing = []
  for k in datasets[1:]:
    train = [[dirPath, fname] for dirPath, _, fname in walk(k)]
    test = [train[0][0] + '/' + train[0][1].pop(-1)]
    training.append(
        [train[0][0] + '/' + p for p in train[0][1] if not p == '.DS_Store'])
    testing.append(test)
  return training, testing


def main():
    projects = [Name for _, Name, __ in walk('./')][0]
    numData = len(projects)  # Number of data
    one, two = explore('./')
    data = [one[i] + two[i] for i in xrange(len(one))]
    set_trace()

if __name__=='__main__':
  main()