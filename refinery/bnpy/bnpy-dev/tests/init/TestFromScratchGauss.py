'''
Unit tests for FromScratchGauss.py
'''
import unittest
import numpy as np
from bnpy.data import XData
from bnpy import HModel
from matplotlib import pylab

class TestFromScratchGauss(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    PRNG = np.random.RandomState(867)
    X = PRNG.randn(100,2)
    self.Data = XData(X=X)

    aPDict = dict(alpha0=1.0)
    oPDict = dict(min_covar=1e-9)
    self.hmodel = HModel.CreateEntireModel('EM','MixModel','ZMGauss', aPDict, oPDict, self.Data)

  def test_viable_init(self, K=7):
    ''' Verify hmodel after init can be used to perform E-step
    '''
    for initname in ['randexamples', 'randsoftpartition', 'randexamplesbydist']:
      initParams = dict(initname=initname, seed=0, K=K)
      self.hmodel.init_global_params(self.Data, **initParams)
      LP = self.hmodel.calc_local_params(self.Data)
      resp = LP['resp']
      assert np.all(np.logical_and(resp >=0, resp <= 1.0))

  def test_consistent_random_seed(self, K=7):
    hmodel = self.hmodel 
    for initname in ['randexamples', 'randsoftpartition', 'randexamplesbydist']:
      initParams = dict(initname=initname, seed=0, K=K)
      hmodel2 = self.hmodel.copy()
      hmodel.init_global_params(self.Data, **initParams)
      hmodel2.init_global_params(self.Data, **initParams)
      assert np.allclose(hmodel.allocModel.w, hmodel2.allocModel.w)
      assert np.allclose(hmodel.obsModel.comp[0].Sigma, hmodel2.obsModel.comp[0].Sigma)
      assert np.allclose(hmodel.obsModel.comp[K-1].Sigma, hmodel2.obsModel.comp[K-1].Sigma)

class TestRandExamplesByDist(unittest.TestCase):
  ''' Create evenly spaced blobs on number line at 0, 1, 2, ...K with small variance
       and make sure that all distinct blobs are chosen at least 95% of time
       when initializing cluster centers.  This verifies that the "plusplus" init works.
  '''
  def setUp(self, K=5):
    self.K = K
    self.MakeData(K=K)
    self.MakeModel()
    
  def MakeModel(self):
    aPDict = dict(alpha0=1.0)
    oPDict = dict(min_covar=1e-9)
    self.hmodel = HModel.CreateEntireModel('EM','MixModel','Gauss', aPDict, oPDict, self.Data)
    
  def MakeData(self, K=5, Nperclass=1000):
    PRNG = np.random.RandomState(867)
    sigma = 1e-3
    Xlist = list()
    for k in range(K):
      Xcur = sigma * PRNG.randn(Nperclass, 2)
      Xcur += k
      Xlist.append(Xcur)
    self.Data = XData(np.vstack(Xlist))
    #pylab.plot(self.Data.X[:,0], self.Data.X[:,1], 'k.')
    #pylab.show()
    
  def test_randexamplesbydist(self, nTrial=25):
    trialOutcomes = list()
    for trial in range(nTrial):
      initParams = dict(initname='randexamplesbydist', seed=trial, K=self.K)
      self.hmodel.init_global_params(self.Data, **initParams)
      muLocs = list()
      for k in range(self.K):
        mu = self.hmodel.obsModel.get_mean_for_comp(k)
        muLocs.append(int(np.round(mu[0])))
      muLocs = np.sort(muLocs)
      nMatch = np.sum( [muLocs[k] == k for k in range(self.K)])
      trialOutcomes.append(nMatch)
    trialOutcomes = np.asarray(trialOutcomes)
    assert np.sum(trialOutcomes == self.K) > 0.95 * nTrial
      