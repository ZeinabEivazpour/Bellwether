#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from os import environ, getcwd
from os import walk
from pdb import set_trace
import sys

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from Prediction import *
from methods1 import *
import dEvol
from stats import ABCD
from texttable import Texttable
from sk import rdivDemo
import logo
#/*Unused references
from abcd import _Abcd
from demos import cmd
import numpy as np
import pandas as pd
import csv
from numpy import sum, mean, std
#*/

def getTunings(fname):
  raw = pd.read_csv('tunings.csv').transpose().values.tolist()
  formatd = pd.DataFrame(raw[1:], columns=raw[0])
  return formatd[fname].values.tolist()

class data():

  """
  Hold training and testing data
  """

  def __init__(self, dataName='ant', dir="./Data/Jureczko"):
    projects = [Name for _, Name, __ in walk(dir)][0]
    numData = len(projects)  # Number of data
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]

    def withinClass(data):
      N = len(data)
      return [(data[:n], [data[n]]) for n in range(1, N)]

    def whereis():
      for indx, name in enumerate(projects):
        if name == dataName:
          return indx

    loc = whereis()

    self.train=[]
    self.test=[]
    for idx, dat in enumerate(data):
      if idx!=loc:
        self.train.append(dat)
      # else:
      #   self.train.append(dat[:-1])
    self.test = data[loc]


class simulate():

  def __init__(self, file='ant', tune=True):
    self.file = file
    self.param = getTunings(file)
    # set_trace()

  def bellwether(self):
    everything=[]
    src = data(dataName=self.file)
    self.test = createTbl(src.test, isBin=True)

    "Pretty Print Thresholds"
    table = Texttable()
    table.set_cols_align(["l","l","l","l"])
    table.set_cols_valign(["m","m","m","m"])
    table.set_cols_dtype(['t', 't', 't', 't'])
    table_rows=[["Dataset", "G2", "Pd", "Pf"]]

    if len(src.train)<9: train=src.train[0]
    else: train=src.train
    header=[" "]
    # onlyMe = [self.file]
    val=[]
    for file in train:

      try: fname = file[0].split('/')[-2]
      except: set_trace()

      header.append(fname)
      self.train = createTbl(file, isBin=True)

      # for _ in xrange(1):
      actual = Bugs(self.test)
      predicted = rforest(
          self.train,
          self.test,
          tunings=self.param,
          smoteit=True)
      p_buggy = [a for a in ABCD(before=actual, after=predicted).all()]
      # val.append(p_buggy[1].stats()[-2])
      val.append([fname, "%0.2f"%p_buggy[1].stats()[-3], "%0.2f"%p_buggy[1].stats()[0], "%0.2f"%p_buggy[1].stats()[1]])
    table_rows.append(sorted(val, key=lambda F: float(F[1])))
    try: table.add_rows(table_rows)
    except: set_trace()
    print(table.draw(), "\n")

    # for a,b in zip(header[1:], onlyMe[1:]):
    #   print(a+'  \t  '+"%0.2f , %0.2f"%(b[0], b[1]))

    # set_trace()
    # everything.append(onlyMe)
    #
    # rdivDemo(everything)

    # ---------- DEBUG ----------
    #   set_trace()

if __name__ == "__main__":
  for file in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
               'lucene', 'poi', 'velocity', 'xalan', 'xerces']:

    print('### ' + file)
    simulate(file).bellwether()
