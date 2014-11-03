'''
Unit tests to verify that our proposed proximity functions work as expected.

Proximity function (defined in Util) are used to determine if two estimated parameters are "close enough" within some numerical tolerance to be treated as equivalent. We eventually use these functions to assess whether learning algorithms like EM are able to estimate "true"/"ideal" parameters for toy data.
'''

import unittest
import numpy as np

import bnpy
import Util

class TestVectorProxFunc(unittest.TestCase):
  def test_vectorproxfunc(self):
    avec = np.asarray([1, 0, 0])
    bvec = np.asarray([0.91, 0.03, 0.08])
    assert np.all( Util.VectorProxFunc(avec, bvec))

    bvec = np.asarray([0.91, -0.03, -0.08])
    assert np.all( Util.VectorProxFunc(avec, bvec))

    bvec = np.asarray([0.99, 0.11, 0.12])
    assert not np.all( Util.VectorProxFunc(avec, bvec))


class TestStarCovarK5(unittest.TestCase):
  ''' Verify CovMatProxFunc discriminates between all StarCovarK5 cov matrices.
  '''

  def setUp(self):
    import StarCovarK5
    self.Sigma = StarCovarK5.Sigma.copy()
    self.SigmaHat = np.zeros_like(self.Sigma)
    for k in range(5):
      Xk = Util.MakeZMGaussData(self.Sigma[k], 10000, seed=k)
      self.SigmaHat[k] = np.cov(Xk.T, bias=1)

  def test_CovMatProxFunc(self):
    print ''
    K = self.Sigma.shape[0]
    for k in xrange(K):
      isG = Util.CovMatProxFunc(self.Sigma[k], self.SigmaHat[k])
      if not np.all(isG):
        Util.pprint( self.Sigma[k], 'true')
        Util.pprint( self.SigmaHat[k], 'est')
        Util.pprint( np.diag(isG).min())
        from IPython import embed; embed()
      assert np.all(isG)
    for k in xrange(K):
      for j in xrange(k+1, K):
        print k,j
        isG = Util.CovMatProxFunc(self.Sigma[k], self.SigmaHat[j])
        if np.all(isG):
          print self.Sigma[k]
          print self.SigmaHat[j]
        assert not np.all(isG)



class TestDeadLeavesD25(TestStarCovarK5):
  ''' Verify CovMatProxFunc discriminates between DeadLeavesD25 cov matrices.
  '''
  def setUp(self):
    import DeadLeavesD25
    self.Sigma = DeadLeavesD25.DL.Sigma.copy()
    self.SigmaHat = np.zeros_like(self.Sigma)
    for k in range(8):
      Xk = Util.MakeZMGaussData(self.Sigma[k], 10000, seed=k)
      self.SigmaHat[k] = np.cov(Xk.T, bias=1)

