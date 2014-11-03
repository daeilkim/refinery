'''
Unit tests for GlobalStickbreakOptimizer.py

Test Strategy:
We examine models with K=2, K=10, and K=128
We start with a known v / beta, and 
generate the "observed data": group-level probs Pi

For each model we verify that both,

1) the neglobprob objective function provided by GSO.py
    has (more or less) a minimum at the known value of v
2) the estimate_v function provided by GSO.py
    can recover that true minimum, or something slightly better.
'''

import sys
import numpy as np
sys.path.append('../../../bnpy/allocmodel/admix/')
import GlobalStickbreakOptimizer as GSO

EPS = 1e-9
    
class TestGSOK2(object):
  def shortDescription(self):
    return None

  def setUp(self):
    '''
    Create a  simple case for making sure we're calculating things correctly
    '''
    self.MakeModel()
    self.MakeData()
    assert self.logPiMat.shape[1] == self.K + 1
    
  def MakeModel(self):
    self.alpha0 = 2
    self.gamma = 50
    self.K = 2
    self.truebeta = np.asarray([0.5, 0.5, 1e-5])
    self.truebeta /= np.sum(self.truebeta)
  
  def MakeData(self):
    self.G = 10000
    PRNG = np.random.RandomState(0)
    Pi = np.zeros( (self.G, self.K+1))
    Pi = PRNG.dirichlet(self.gamma * self.truebeta, G)
    Pi = np.maximum(Pi, EPS)
    self.logPiMat = np.log(Pi)
    self.PRNG = PRNG
    
  def test_estimate_is_near_truth(self, nTrial=100):
    vopt = GSO.beta2v(self.truebeta)
    objfunc = lambda v: GSO.neglogp(v, self.G, self.logPiMat, self.alpha0, self.gamma)
    success = 0
    print vopt, '*optimal*'
    for trial in range(nTrial):
      v = GSO.estimate_v(self.G, self.logPiMat, self.alpha0, self.gamma, method='tnc')
      if np.all( np.abs(v - vopt) < .01 ):
        success += 1
      elif objfunc(v) < objfunc(vopt):
        success += 1
      else:
        print v
    print "%d/%d suceeded." % (success, nTrial)
    assert success == nTrial

  def test_truth_is_minimum_of_objfunc(self, nTrial=100):
    vopt = GSO.beta2v(self.truebeta)
    objfunc = lambda v: GSO.neglogp(v, self.G, self.logPiMat, self.alpha0, self.gamma)
    success = 0
    fopt = objfunc(vopt)
    print vopt, fopt, '**'

    for trial in range(nTrial):
      v = vopt + 0.01 * self.PRNG.rand(self.K)
      v = np.minimum(v, 1.0 - 1e-8)
      fv = objfunc(v)
      if fopt < fv:
        success += 1
      else:
        print v, fv
    assert success > 0.98 * nTrial
    
######################################################### K= 10
######################################################### 
class TestGSOK10(TestGSOK2):   
  def MakeModel(self):
    self.alpha0 = 2
    self.gamma = 50
    self.K = 10
    self.truebeta = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-5])
    self.truebeta /= np.sum(self.truebeta)

######################################################### K= 128
######################################################### 
class TestGSOK128(TestGSOK2):   
  def MakeModel(self):
    self.alpha0 = 2
    self.gamma = 50
    self.K = 128
    self.truebeta = np.hstack( [np.ones(self.K), 1e-5])
    self.truebeta /= np.sum(self.truebeta)

