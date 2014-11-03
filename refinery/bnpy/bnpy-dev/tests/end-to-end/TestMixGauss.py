'''
Unit-tests for full learning for full-mean, full-covariance Gaussian models
'''
import numpy as np
import unittest

import bnpy
from AbstractEndToEndTest import AbstractEndToEndTest
import Util

class TestSimple(AbstractEndToEndTest):
  ''' Test basic functionality (run without crashing?) on very simple dataset. 
  '''
  __test__ = True

  def setUp(self):
    PRNG = np.random.RandomState(333)
    X = PRNG.randn(1000, 3)
    self.Data = bnpy.data.XData(X)
    self.allocModelName = 'MixModel'
    self.obsModelName = 'Gauss'  
    self.kwargs = dict(nLap=30, K=3, alpha0=1)
    self.kwargs['smatname'] = 'eye'

    self.learnAlgs = ['EM', 'VB', 'moVB', 'soVB']


class TestEasyK2_EM(AbstractEndToEndTest):
  ''' Test basic EM parameter estimation on well-separable K=2 toy dataset.

      Verify runs from fromTruth and fromScratch inits reach ideal params.
  '''
  __test__ = True

  def setUp(self):
    # Define true parameters (mean, prec matrix) for 2 well-separated clusters
    self.K = 2
    B = 20
    Mu = np.eye(2)
    Sigma = np.zeros((2,2,2))
    Sigma[0] = np.asarray([[B,0], [0,1./B]])
    Sigma[1] = np.asarray([[1./B,0], [0,B]])    
    L = np.zeros_like(Sigma)
    for k in xrange(self.K):
      L[k] = np.linalg.inv(Sigma[k])    
    self.TrueParams = dict(w=0.5*np.ones(self.K), K=self.K, m=Mu, L=L)
    self.ProxFunc = dict(L=Util.CovMatProxFunc,
                         m=Util.VectorProxFunc,
                         w=Util.ProbVectorProxFunc)

    # Generate data
    Nk = 1000
    X = Util.MakeGaussData(Mu, Sigma, Nk)
    self.Data = bnpy.data.XData(X)

    self.learnAlgs = ['EM']

    # Basic configuration
    self.allocModelName = 'MixModel'
    self.obsModelName = 'Gauss'  
    self.kwargs = dict(nLap=30, K=self.K, alpha0=1.0)
    
    # Substitute config used for "from-scratch" tests only
    #  anything in here overrides defaults in self.kwargs
    self.fromScratchArgs = dict(nLap=50, K=self.K, initname='randexamples')
