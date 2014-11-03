'''
Unit tests for FromScratchGauss.py
'''
import unittest
import numpy as np
from bnpy.data import XData
from bnpy import HModel
from bnpy.ioutil import ModelWriter, ModelReader

class TestFromScratchGauss(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self, K=7):
    ''' Create random data, and a K component MixModel to go with it
        Call this original model "hmodel".
        We copy hmodel into "modelB", and then save to file via save_model()
    '''
    self.K = K
    PRNG = np.random.RandomState(867)
    X = PRNG.randn(100,2)
    self.Data = XData(X=X)

    aPDict = dict(alpha0=1.0)
    oPDict = dict(min_covar=1e-9)
    self.hmodel = HModel.CreateEntireModel('EM','MixModel','ZMGauss', 
                                            aPDict, oPDict, self.Data)
    modelB = self.hmodel.copy()    
    initParams = dict(initname='randexamples', seed=0, K=self.K)
    modelB.init_global_params(self.Data, **initParams)
    ModelWriter.save_model(modelB, '/tmp/', 'Test')
    self.modelB = modelB

  def test_viable_init(self):
    ''' Verify hmodel after init can be used to perform E-step
    '''
    initSavedParams = dict(initname='/tmp/', prefix='Test')
    self.hmodel.init_global_params(self.Data, **initSavedParams)
    assert self.hmodel.allocModel.K == self.K
    keysA = self.hmodel.allocModel.to_dict()
    keysB = self.modelB.allocModel.to_dict()
    assert len(keysA) == len(keysB)
    
    aLP = self.hmodel.calc_local_params(self.Data)
    assert np.all(np.logical_and(aLP['resp']>=0,aLP['resp']<=1.0))
    assert np.allclose(1.0, np.sum(aLP['resp'],axis=1))


