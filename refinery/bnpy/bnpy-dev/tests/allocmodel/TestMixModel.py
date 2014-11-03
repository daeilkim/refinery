'''
Unit-tests for MixModel.py
'''
import numpy as np
import bnpy
from bnpy.allocmodel import MixModel
from bnpy.suffstats import SuffStatBag

class TestMixModelEMUnifAlpha(object):
  def shortDescription(self):
    return None

  def setUp(self):
    '''
    Create a stupid simple case for making sure we're calculating things correctly
    '''
    self.alpha0 = 1.0
    self.allocM = MixModel('EM', dict(alpha0=self.alpha0))
    self.N = np.asarray([1.,2.,3,4,5.])
    self.SS = SuffStatBag(K=5, D=1)
    self.SS.setField('N', self.N, dims='K')
    self.resp = np.random.rand(100,3)
    self.precompEntropy = np.sum(self.resp * np.log(self.resp), axis=0)
    
  def test_update_global_params_EM(self):
    self.allocM.update_global_params_EM(self.SS)
    wTrue = (self.N + self.alpha0 - 1.0)
    wTrue = wTrue / np.sum(wTrue)
    wEst = self.allocM.w
    print wTrue
    print wEst
    assert np.allclose(wTrue, wEst)
    
  def test_get_global_suff_stats(self):
    Data = bnpy.data.XData(np.random.randn(10,1))
    SS = self.allocM.get_global_suff_stats(Data, dict(resp=self.resp), doPrecompEntropy=True)
    assert np.allclose(self.precompEntropy, SS.getELBOTerm('ElogqZ'))
    assert np.allclose( np.sum(self.resp, axis=0), SS.N)

class TestMixModelEMNonunifAlpha(TestMixModelEMUnifAlpha):
  def setUp(self):
    self.alpha0 = 2.0
    self.allocM = MixModel('EM', dict(alpha0=self.alpha0))
    self.N = np.asarray([1.,2.,3,4,5.])
    self.SS = SuffStatBag(K=5, D=1)
    self.SS.setField('N', self.N, dims='K')
    self.resp = np.random.rand(100,3)
    self.precompEntropy = np.sum(self.resp * np.log(self.resp), axis=0)
