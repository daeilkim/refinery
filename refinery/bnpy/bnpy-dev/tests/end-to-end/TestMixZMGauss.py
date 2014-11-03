'''
Unit-tests for full learning for zero-mean, full-covariance Gaussian models
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
    self.obsModelName = 'ZMGauss'  
    self.kwargs = dict(nLap=30, K=4, alpha0=1.0)
    self.learnAlgs = ['EM', 'VB', 'moVB', 'soVB']

class TestEasyK2_EM(AbstractEndToEndTest):
  ''' Test basic EM parameter estimation on well-separable K=2 toy dataset.

      Verify runs from fromTruth and fromScratch inits reach ideal params.
  '''
  __test__ = True

  def setUp(self):
    # Define true parameters: two very-different covariance matrices
    self.K = 2
    B = 20
    Sigma = np.zeros((2,2,2))
    Sigma[0] = np.asarray([[B,0], [0,1./B]])
    Sigma[1] = np.asarray([[1./B,0], [0,B]])    
    self.TrueParams = dict(Sigma=Sigma, w=0.5*np.ones(self.K))

    # Functions used by tests to decide if estimated params are "close enough"
    # Must have same keys as self.TrueParams
    self.ProxFunc = dict(Sigma=Util.CovMatProxFunc,
                          w=Util.ProbVectorProxFunc)

    # Generate toy dataset
    Nk = 1000
    X = Util.MakeZMGaussData(Sigma, Nk, seed=34567)
    self.Data = bnpy.data.XData(X)

    # Only run EM tests
    self.learnAlgs = ['EM']

    # Basic model configuration
    self.allocModelName = 'MixModel'
    self.obsModelName = 'ZMGauss'  
    self.kwargs = dict(nLap=30, K=self.K, alpha0=1.0)

    # Substitute config used for "from-scratch" tests only
    #  anything in here overrides defaults in self.kwargs
    self.fromScratchArgs = dict(nLap=50, K=self.K, initname='randexamples')
    self.fromScratchTrials = 5
    self.fromScratchSuccessRate = 0.5


class TestStarCovarK5_EM(AbstractEndToEndTest):
  ''' Test basic EM parameter estimation on StarCovarK5 toy dataset.

      Verify runs from fromTruth and fromScratch inits estimate ideal params.
  '''
  __test__ = True

  def setUp(self):
    self.K = 5
    import StarCovarK5
    self.Data = StarCovarK5.get_data(nObsTotal=10000)
    
    self.TrueParams = dict(Sigma=StarCovarK5.Sigma,
                           w=StarCovarK5.w)
    self.ProxFunc = dict(Sigma=Util.CovMatProxFunc,
                          w=Util.ProbVectorProxFunc)

    self.learnAlgs = ['EM']

    self.allocModelName = 'MixModel'
    self.obsModelName = 'ZMGauss'  
    self.kwargs = dict(nLap=30, K=self.K, alpha0=1.0)

    self.fromScratchArgs = dict(nLap=50, K=self.K, initname='randexamples')
    self.fromScratchTrials = 10
    self.fromScratchSuccessRate = 0.5