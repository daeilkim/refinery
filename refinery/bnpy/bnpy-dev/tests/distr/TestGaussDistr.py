'''
Unit tests for GaussDistr.py
'''
from bnpy.distr import GaussDistr
import numpy as np

class TestGaussD2(object):
  def setUp(self):
    self.m = np.ones(2)
    self.invSigma = np.eye(2)
    self.distr = GaussDistr(m=self.m, L=self.invSigma)
    
  def test_dimension(self):
    assert self.distr.D == self.invSigma.shape[0]
    
  def test_cholL(self):
    chol = self.distr.cholL()
    assert np.allclose(np.dot(chol, chol.T), self.distr.L)
    
  def test_logdetL(self):
    logdetL = self.distr.logdetL()
    assert np.allclose( np.log(np.linalg.det(self.invSigma)), logdetL)
  
  def test_dist_mahalanobis(self, N=10):
    X = np.random.randn(N, self.distr.D)
    Dist = self.distr.dist_mahalanobis(X)
    invSigma = self.invSigma
    MyDist = np.zeros(N)
    for ii in range(N):
      x = X[ii] - self.m
      MyDist[ii] = np.dot(x.T, np.dot(invSigma, x))
      #if error, we print it out
      print MyDist[ii], Dist[ii]
    assert np.allclose(MyDist, Dist)
    
class TestGaussD1(TestGaussD2):
  def setUp(self):
    self.m = np.ones(1)
    self.invSigma = np.eye(1)
    self.distr = GaussDistr(m=self.m, L=self.invSigma)
    
    
class TestGaussD10(TestGaussD2):
  def setUp(self):
    PRNG = np.random.RandomState(867)
    R = PRNG.rand(10,10)

    self.m = np.ones(10)
    self.invSigma = 1e-4*np.eye(10)
    self.distr = GaussDistr(m=self.m, L=self.invSigma)
    