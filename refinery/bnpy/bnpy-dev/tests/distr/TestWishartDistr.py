'''
'''
from bnpy.distr import WishartDistr
from bnpy.suffstats import SuffStatBag
import numpy as np
import copy

class TestWishart(object):
  def setUp(self):
    self.v = 4
    self.invW = np.eye(2)
    self.distr = WishartDistr(v=self.v, invW=self.invW)
    
  def test_dimension(self):
    assert self.distr.D == self.invW.shape[0]
    
  def test_cholinvW(self):
    cholinvW = self.distr.cholinvW()
    assert np.allclose(np.dot(cholinvW, cholinvW.T), self.distr.invW)
  
  def test_expected_covariance_matrix(self):
    CovMat = self.distr.ECovMat()
    MyCovMat = self.invW / (self.v - self.distr.D - 1)
    print MyCovMat, CovMat
    assert np.allclose(MyCovMat, CovMat)
    
  def test_post_update_soVB(self, rho=0.375):
    distrA = copy.deepcopy(self.distr)
    distrB = WishartDistr(distrA.v, invW=np.eye(distrA.D) )        
    self.distr.post_update_soVB(rho, distrB)
    assert self.distr.v == rho*distrA.v + (1-rho)*distrB.v
    assert np.allclose(self.distr.invW, rho*distrA.invW + (1-rho)*distrB.invW)
    
  def test_entropy_posterior_gets_smaller(self, N=1):
    PRNG = np.random.RandomState(seed=8675309)
    for trial in range(3):
      X = PRNG.randn(N, self.distr.D)
      xxT = np.dot(X.T, X)

      SS = SuffStatBag(K=1, D=self.distr.D)
      SS.setField('N', [N], dims='K')
      SS.setField('xxT', [xxT], dims=('K','D','D'))

      postD = self.distr.get_post_distr(SS, 0)
      assert postD.D == self.distr.D
      Hpost = postD.get_entropy()
      Hprior = self.distr.get_entropy()
      print 'Prior %.3g, Post %.3g' % (Hprior, Hpost)
      print self.distr.invW
      print postD.invW
      assert Hpost < Hprior