'''
'''
from bnpy.distr import ZMGaussDistr
import numpy as np

class TestZMGauss(object):
  def setUp(self):
    self.Sigma = np.eye(4)
    self.distr = ZMGaussDistr(Sigma=self.Sigma.copy())
    
  def test_dimension(self):
    assert self.distr.D == self.Sigma.shape[0]
    
  def test_cholSigma(self):
    chol = self.distr.cholSigma()
    assert np.allclose(np.dot(chol, chol.T), self.distr.Sigma)
    
  def test_logdetSigma(self):
    logdetSigma = self.distr.logdetSigma()
    assert np.allclose( np.log(np.linalg.det(self.Sigma)), logdetSigma)
  
  def test_get_log_norm_const(self):
    logZ = self.distr.get_log_norm_const()
    logdetSigma = np.log(np.linalg.det(self.Sigma))
    mylogZ = 0.5*self.Sigma.shape[0]*np.log(2*np.pi) + 0.5 * logdetSigma
    
  def test_dist_mahalanobis(self, N=10):
    X = np.random.randn(N, self.distr.D)
    Dist = self.distr.dist_mahalanobis(X)
    invSigma = np.linalg.inv(self.Sigma)
    MyDist = np.zeros(N)
    for ii in range(N):
      x = X[ii]
      MyDist[ii] = np.dot(x.T, np.dot(invSigma, x))
      #if error, we print it out
      print MyDist[ii], Dist[ii]
    assert np.allclose(MyDist, Dist)

class TestZMGaussRand1Dim(TestZMGauss):
  def setUp(self):
    self.Sigma = np.asarray([[42.0]])
    self.distr = ZMGaussDistr(Sigma=self.Sigma)
  
class TestZMGaussRand5Dim(TestZMGauss):
  def setUp(self):
    R = np.random.rand(5,5)
    self.Sigma = np.dot(R, R.T) + 0.02*np.eye(5)
    self.distr = ZMGaussDistr(Sigma=self.Sigma)