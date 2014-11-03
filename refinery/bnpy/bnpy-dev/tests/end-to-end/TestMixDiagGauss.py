'''
'''
import numpy as np
import unittest

import bnpy
from AbstractEndToEndTest import AbstractEndToEndTest
import Util

class TestSimple(AbstractEndToEndTest):
  __test__ = True

  def setUp(self):    
    PRNG = np.random.RandomState(333)
    X = PRNG.randn(1000, 3)
    self.Data = bnpy.data.XData(X)
    self.allocModelName = 'MixModel'
    self.obsModelName = 'DiagGauss'  
    self.kwargs = dict(nLap=30, K=3, alpha0=1)
    self.learnAlgs = ['EM', 'VB', 'moVB', 'soVB']

