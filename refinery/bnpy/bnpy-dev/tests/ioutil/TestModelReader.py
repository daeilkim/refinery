'''
Unit tests for ModelReader.py
'''
from bnpy.ioutil import ModelWriter, ModelReader
from bnpy import HModel
from bnpy.data import XData
import numpy as np
import unittest

class TestModelReaderEMK1(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    PRNG = np.random.RandomState(867)
    X = PRNG.randn(100,2)
    self.Data = XData(X=X)

    aPDict = dict(alpha0=1.0)
    oPDict = dict(min_covar=1e-9)
    self.hmodel = HModel.CreateEntireModel('EM','MixModel','ZMGauss', aPDict, oPDict, self.Data)
    initParams = dict(initname='randexamples', seed=0, K=1)
    self.hmodel.init_global_params(self.Data, **initParams)

  def test_save_then_load_same_model(self, prefix='Test'):
    ''' Verify that we can save model H to disk, load it back in as G, and 
        get same results from calculations on H,G with same inputs
    '''
    ModelWriter.save_model(self.hmodel, '/tmp/', prefix)
    gmodel = ModelReader.load_model('/tmp/', prefix)
    assert type(gmodel.allocModel) == type(self.hmodel.allocModel)
    assert gmodel.allocModel.K == self.hmodel.allocModel.K
    K = gmodel.allocModel.K
    for k in range(K):
      gSig = gmodel.obsModel.comp[k].ECovMat()
      hSig = self.hmodel.obsModel.comp[k].ECovMat()
      print gSig, hSig
      assert np.allclose(gSig, hSig)

    # Make sure we get same responsibilities before and after
    hLP = self.hmodel.calc_local_params(self.Data)
    gLP = gmodel.calc_local_params(self.Data)
    assert np.allclose(hLP['resp'], gLP['resp'])
    

class TestModelReaderEMMixZM(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    PRNG = np.random.RandomState(867)
    X = PRNG.randn(100,2)
    self.Data = XData(X=X)

    aPDict = dict(alpha0=1.0)
    oPDict = dict(min_covar=1e-9)
    self.hmodel = HModel.CreateEntireModel('EM','MixModel','ZMGauss', aPDict, oPDict, self.Data)
    initParams = dict(initname='randexamples', seed=0, K=5)
    self.hmodel.init_global_params(self.Data, **initParams)

  def test_save_then_load_same_model(self, prefix='Test'):
    ''' Verify that we can save model H to disk, load it back in as G, and 
        get same results from calculations on H,G with same inputs
    '''
    ModelWriter.save_model(self.hmodel, '/tmp/', prefix)
    gmodel = ModelReader.load_model('/tmp/', prefix)
    assert type(gmodel.allocModel) == type(self.hmodel.allocModel)
    assert gmodel.allocModel.K == self.hmodel.allocModel.K
    K = gmodel.allocModel.K
    for k in range(K):
      gSig = gmodel.obsModel.comp[k].ECovMat()
      hSig = self.hmodel.obsModel.comp[k].ECovMat()
      assert np.allclose(gSig, hSig)

    # Make sure we get same responsibilities before and after
    hLP = self.hmodel.calc_local_params(self.Data)
    gLP = gmodel.calc_local_params(self.Data)
    assert np.allclose(hLP['resp'], gLP['resp'])
    
    
class TestModelReaderVBMixZM(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    PRNG = np.random.RandomState(867)
    X = PRNG.randn(100,2)
    self.Data = XData(X=X)

    aPDict = dict(alpha0=1.0)
    oPDict = dict(dF=4.0, ECovMat='eye', sF=1.0)

    self.hmodel = HModel.CreateEntireModel('VB','MixModel','ZMGauss', aPDict, oPDict, self.Data)
    initParams = dict(initname='randexamples', seed=0, K=5)
    self.hmodel.init_global_params(self.Data, **initParams)

  def test_save_then_load_same_model(self, prefix='Test'):
    ''' Verify that we can save model H to disk, load it back in as G, and 
        get same results from calculations on H,G with same inputs
    '''
    ModelWriter.save_model(self.hmodel, '/tmp/', prefix)
    gmodel = ModelReader.load_model('/tmp/', prefix)
    assert type(gmodel.allocModel) == type(self.hmodel.allocModel)
    assert gmodel.allocModel.K == self.hmodel.allocModel.K
    K = gmodel.allocModel.K
    for k in range(K):
      gSig = gmodel.obsModel.comp[k].ECovMat()
      hSig = self.hmodel.obsModel.comp[k].ECovMat()
      assert np.allclose(gSig, hSig)

    # Make sure we get same responsibilities before and after
    hLP = self.hmodel.calc_local_params(self.Data)
    gLP = gmodel.calc_local_params(self.Data)
    assert np.allclose(hLP['resp'], gLP['resp'])
    

