'''
Unit tests for HDPBetaOptimizer.py

Test Strategy:
We examine models with K=2
We start with a known v / beta, and 
generate the "observed data": group-level probs Pi

For each model we verify that both,

1) the objectiveFunc objective function provided by HDPBetaOptimizer.py
    has (more or less) a minimum at the known value of v
2) the estimate_v function provided by HDPBetaOptimizer.py
    can recover that true minimum, or something slightly better.
'''

import sys
import unittest
import numpy as np
from scipy.special import gammaln, digamma, polygamma
sys.path.append('../../../bnpy/allocmodel/admix/')
import HDPBetaOptimizer as HBO

EPS = 1e-9
    
class TestHBOK2(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    '''
    Create a  simple case for making sure we're calculating things correctly
    '''
    self.MakeModel()
    self.MakeData()
    assert self.sumLogPi.shape[1] == self.K + 1
    
  def MakeModel(self):
    self.alpha = .5
    self.gamma = 5
    self.K = 2
    self.truebeta = np.asarray([0.5, 0.5, 1e-2])
    self.truebeta /= np.sum(self.truebeta)

  # Used only if we actually calculate the variational expectations
  def calcExpPI(self, Pi):
    ELogPi = np.zeros((self.G, self.K+1))
    for i in xrange(self.G):
       ELogPi[i] = digamma(Pi[i,:]) - digamma(np.sum(Pi[i,:]))
    return ELogPi

  def MakeData(self):
    self.G = 1000
    PRNG = np.random.RandomState(0)
    Pi = np.zeros( (self.G, self.K+1) )
    Pi = PRNG.dirichlet(self.gamma * self.truebeta, self.G)
    Pi = np.maximum(Pi, EPS)
    self.sumLogPi = np.log(Pi)
    #ElogPi = self.calcExpPI(Pi)
    #self.sumLogPi = np.asarray(np.sum(ElogPi, axis=0))
    self.PRNG = PRNG

  def test_estimate_is_near_truth(self, nTrial=2):
    vopt = HBO.beta2v(self.truebeta)
    objfunc = lambda v: HBO.objectiveFunc(v, self.alpha, self.gamma, self.G, self.sumLogPi)
    success = 0
    print vopt, '*optimal*'

    for trial in range(nTrial):
      initV = np.random.rand(self.K)
      v = HBO.estimate_v(self.gamma, self.alpha, self.G, self.K, self.sumLogPi, initV )
      if np.all( np.abs(v - vopt) < .01 ):
        success += 1
      elif objfunc(v) < objfunc(vopt):
        success += 1
      else:
        print v
    print "%d/%d suceeded." % (success, nTrial)
    assert success == nTrial

  def test_truth_is_minimum_of_objfunc(self, nTrial=2):
    vopt = HBO.beta2v(self.truebeta)
    objfunc = lambda v: HBO.objectiveFunc(v, self.alpha, self.gamma, self.G, self.sumLogPi)
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
    


