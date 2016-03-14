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
from sk import rdivDemo
import logo
#/*Unused references
from abcd import _Abcd
from demos import cmd
import numpy as np
import pandas as pd
import csv
from numpy import sum
#*/


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

  def __init__(self, file='ant', tune=False):
    self.file = file
    self.param = dEvol.tuner(rforest, data(dataName=self.file).train[-1]) if \
        tune else None

  def bellwether(self):
    everything=[]
    src = data(dataName=self.file)
    self.test = createTbl(src.test, isBin=True)

    if len(src.train)<12: train=src.train[0]
    else: train=src.train
    for file in train:

      try: fname = file[0].split('/')[-2]
      except: set_trace()

      self.train = createTbl(file, isBin=True)
      onlyMe = [self.file+'-'+fname]

      for _ in xrange(10):
        actual = Bugs(self.test)
        predicted = rforest(
            self.train,
            self.test,
            tunings=self.param,
            smoteit=True)

        p_buggy = [a for a in ABCD(before=actual, after=predicted).all()]
        onlyMe.append(p_buggy[1].stats()[-1])

      everything.append(onlyMe)

    rdivDemo(everything)

    # ---------- DEBUG ----------
    #   set_trace()

if __name__ == "__main__":
  for file in ['camel', 'ant', 'ivy',
               'jedit', 'log4j', 'pbeans',
               'lucene', 'poi', 'synapse',
               'velocity', 'xalan', 'xerces']:
    print('### ' + file)
    simulate(file).bellwether()
