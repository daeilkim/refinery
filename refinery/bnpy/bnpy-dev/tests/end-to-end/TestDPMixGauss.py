'''
Unit-tests for full learning for full-mean, full-covariance Gaussian models
'''
import numpy as np
import unittest

import bnpy
from AbstractEndToEndTest import AbstractEndToEndTest
import Util

class TestDPMixGaussModel(AbstractEndToEndTest):
  __test__ = True

  def setUp(self):
    PRNG = np.random.RandomState(333)
    X = PRNG.randn(1000, 3)
    X = np.vstack([X, 5 + PRNG.randn(100, 3)])
    self.Data = bnpy.data.XData(X)
    self.allocModelName = 'DPMixModel'
    self.obsModelName = 'Gauss'  
    self.kwargs = dict(nLap=30, K=5, alpha0=1)
    self.kwargs['smatname'] = 'eye'

    self.learnAlgs = ['VB', 'soVB', 'moVB']
