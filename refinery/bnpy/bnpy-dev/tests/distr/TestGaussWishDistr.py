'''
Unit tests for GaussWishDistr.py
'''
from bnpy.distr import GaussWishDistr, WishartDistr
from bnpy.suffstats import SuffStatBag
import numpy as np
import copy

class TestGaussWishDistr(object):
  def setUp(self):
    self.invW = np.eye(2)
    self.m = np.zeros(2)
    self.distr = GaussWishDistr(m=self.m, invW=self.invW, kappa=1.0, dF=4)
    
  def test_dimension(self):
    assert self.distr.D == self.invW.shape[0]
    
  def test_entropyWish(self):
    ''' Verify that (wishart) entropy is same for this object and Wishart object 
    '''
    Hself = self.distr.entropyWish()
    wishDistr = WishartDistr(v=self.distr.dF, invW=self.distr.invW)
    Hwish = wishDistr.get_entropy()
    assert np.allclose( Hself, Hwish)
    
  def test_dist_mahalanobis(self, N=10):
    ''' Verify that distance computation is largest at mean and decays further away 
    '''
    Xlist = list()
    for r in [0, 0.01, 0.1, 1, 2, 3, 4, 5]:
      Xlist.append(self.distr.m + r)
    X = np.asarray(Xlist)
    Dist = self.distr.dist_mahalanobis(X)
    print Dist
    assert np.all( Dist[:-1] < Dist[1:])
    
  def test_update_soVB(self, rho=0.25):
    ''' Verify the blend update for stochastic variational is correct
    '''
    distrB = copy.deepcopy(self.distr)
    distrB.invW *= 3
    distrB.m += 2
    distrB.kappa *= 10
    distrB2 = copy.deepcopy(distrB)
    # Make sure things are different!
    assert not np.allclose(distrB.invW, self.distr.invW)
    assert not np.allclose(distrB.m, self.distr.m)
    
    distrB.post_update_soVB(rho, self.distr)
    assert distrB.dF == distrB2.dF * (1-rho) + self.distr.dF * rho
    assert np.allclose(distrB.kappa, distrB2.kappa * (1-rho) + self.distr.kappa * rho)

    # these dont work because the parameterization is a bit trickier here.
    #assert np.allclose(distrB.invW, distrB2.invW * (1-rho) + self.distr.invW * rho)
    #assert np.allclose(distrB.m, distrB2.m * (1-rho) + self.distr.m * rho)

    
  def test_entropy_posterior_gets_smaller(self, N=10):
    PRNG = np.random.RandomState(seed=8675309)
    for trial in range(3):
      X = PRNG.randn(N, self.distr.D) + self.distr.m
      x = np.sum(X,axis=0)
      xxT = np.dot(X.T,X)
      SS = SuffStatBag(K=1, D=self.distr.D)
      SS.setField('N', [N], dims='K')
      SS.setField('x', [x], dims=('K','D'))
      SS.setField('xxT', [xxT], dims=('K','D','D'))
      postD = self.distr.get_post_distr(SS, 0)
      assert postD.D == self.distr.D
      Hpost = postD.entropyWish()
      Hprior = self.distr.entropyWish()
      print 'Prior %.3g, Post %.3g' % (Hprior, Hpost)
      assert Hpost < Hprior