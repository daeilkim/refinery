'''
Unit tests for HDPVariationalOptimizer.py (aka HVO)

Test Strategy:
We examine models with K=2, K=10, and K=128
We start with a known v / beta, and 
generate the "observed data": group-level probs Pi

For each model we verify that both,

1) the objective function provided by HVO
    has (more or less) a minimum at the known value of v
2) the estimate_u function provided by HVO
    can recover that true minimum, or something slightly better.
'''

import sys
import numpy as np
import bnpy.allocmodel.admix.HDPVariationalOptimizer as HVO
import unittest

EPS = 1e-12
nDoc=4000

class TestHVOK2(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    '''
    Create a model and some test data for quick experiments.
    '''
    self.MakeModel()
    self.MakeData()
    assert self.Pi.shape[1] == self.K + 1

  def MakeModel(self):
    self.alpha0 = 1.0
    self.gamma = 0.99
    self.K = 2
    self.truebeta = np.asarray([0.5, 0.495, 0.005])
    self.truebeta /= np.sum(self.truebeta)
  
  def MakeData(self):
    self.nDoc = nDoc
    PRNG = np.random.RandomState(0)
    self.Pi = PRNG.dirichlet(self.gamma * self.truebeta, size=self.nDoc)
    self.Pi = np.maximum(self.Pi, 1e-20)
    assert np.allclose(np.sum(self.Pi, axis=1), 1.0)
    self.PRNG = PRNG
    
  def test_vars_defined(self):
    varDict = vars(self)
    assert 'Pi' in varDict
    assert 'gamma' in varDict

  def test_estimate_is_near_truth(self, nTrial=1):
    ''' Verify that we recover variational parameters
          whose E[beta] is very close to the true beta
    '''
    success = 0
    np.set_printoptions(precision=3, suppress=True)
    print self.truebeta, '*optimal*'
    for trial in range(nTrial):
      U1, U0 = HVO.estimate_u(**vars(self))
      Ev = U1 / (U1 + U0)
      Ebeta = HVO.v2beta(Ev)
      if np.all(np.abs(Ebeta - self.truebeta) < .024):
        success += 1
      else:
        print U1
        print U0
        print Ebeta
    print "%d/%d succeeded." % (success, nTrial)
    assert success == nTrial

    

######################################################### K= 10
######################################################### 
class TestHVOK10(TestHVOK2):   
  def MakeModel(self):
    self.alpha0 = 1.0
    self.gamma = 0.99 # closer to 1 makes variance smaller
    self.K = 10
    self.truebeta = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-5])
    self.truebeta /= np.sum(self.truebeta)


######################################################### K= 128
######################################################### 
class TestHVOK128(TestHVOK2):   
  def MakeModel(self):
    self.alpha0 = 1.0
    self.gamma = 0.99
    self.K = 128
    self.truebeta = np.hstack( [np.ones(self.K), 1e-5])
    self.truebeta /= np.sum(self.truebeta)


######################################################### K= 300
######################################################### 
class TestHVOK300(TestHVOK2):   
  def MakeModel(self):
    self.alpha0 = 1.0
    self.gamma = 0.99
    self.K = 300
    self.truebeta = np.hstack( [np.ones(self.K), 1e-5])
    self.truebeta /= np.sum(self.truebeta)

class TestHVOK300nonuniform(TestHVOK2):   
  def MakeModel(self):
    self.alpha0 = 1.0
    self.gamma = 0.99
    self.K = 300
    self.truebeta = np.hstack( [np.random.rand(self.K), 1e-5])
    self.truebeta /= np.sum(self.truebeta)

