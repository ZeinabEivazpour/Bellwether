"""
NOT SURE IF THIS IS NEEDED YET. DEV ON HOLD ATM.
"""
from __future__ import print_function, division
import os
import sys
import jnius_config
import py4j as java
from jnius import autoclass

root = os.path.join(os.getcwd().split('HDP')[0], 'HDP')
if root not in sys.path:
    sys.path.append(root)

jnius_config.add_options('-Xrs', '-Xmx4096m')
jnius_config.set_classpath(
    '.',
    './jars/weka.jar',
    './jars/commons-math3-3.5.jar')

