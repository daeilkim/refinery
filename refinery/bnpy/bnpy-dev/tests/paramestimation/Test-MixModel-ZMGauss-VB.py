'''
Unit tests for basic VB learning for the Mix - ZMGauss hmodel 
'''
import numpy as np
np.set_printoptions(precision=3)

from bnpy import HModel
from bnpy.data import XData
from bnpy.util import RandUtil

########################################## basic tests for 1-cluster model
class TestMixZMVB(object):
  def setUp(self):
    X = np.random.randn(100, 3)
    self.Data = XData(X=X)
    aPDict = dict(alpha0=1.0)
    oPDict = dict(dF=5, ECovMat='eye', sF=1.0)
    self.hmodel = HModel.CreateEntireModel('VB', 'MixModel', 'ZMGauss', aPDict, oPDict, self.Data)
    
  def test_dimension(self):
    assert self.hmodel.obsModel.D == self.Data.dim

  def test_get_global_suff_stats_one_cluster(self):
    ''' Verify correctness when all data forced to be in one cluster by responsibility
    '''
    LP = dict(resp=np.ones((self.Data.nObs,1)))
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    assert np.sum(SS.N) == self.Data.nObs
    assert np.allclose(SS.getComp(0).xxT, np.dot(self.Data.X.T, self.Data.X))
    self.hmodel.update_global_params(SS)
    assert np.allclose( self.hmodel.allocModel.Elogw, [0])


########################################## basic tests for 4-cluster model
class TestMixZMVB_4Class2D(object):
  def setUp(self):
    self.MakeData()
    self.MakeHModel()
  
  def MakeHModel(self):
    aPDict = dict(alpha0=1.0)
    oPDict = dict(dF=5, ECovMat='eye', sF=1.0)
    self.hmodel = HModel.CreateEntireModel('VB', 'MixModel', 'ZMGauss', aPDict, oPDict, self.Data)
  
  def MakeData(self, N=10000):
    S1 = np.asarray([[100, 0], [0, 0.01]])
    Sigma = np.zeros( (2,2,4))
    Sigma[:,:,0] = S1
    Sigma[:,:,1] = RandUtil.rotateCovMat(S1, theta=np.pi/4)
    Sigma[:,:,2] = RandUtil.rotateCovMat(S1, theta=2*np.pi/4)
    Sigma[:,:,3] = RandUtil.rotateCovMat(S1, theta=3*np.pi/4)
    self.Sigma = Sigma
    Xlist = list()
    Rlist = list()
    for k in range(Sigma.shape[2]):
      curX = RandUtil.mvnrand([0,0], Sigma[:,:,k], N)
      curresp = np.zeros((N,4))
      curresp[:,k] = 1.0
      Xlist.append(curX)
      Rlist.append(curresp)
    X = np.vstack(Xlist)
    self.Data = XData(X=X)
    self.trueresp = np.vstack(Rlist)
    
  def test_get_global_suff_stats(self):
    LP = dict(resp=self.trueresp)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    assert np.allclose(SS.N, np.sum(self.trueresp,axis=0))
    
  def test_calc_local_params(self):  
    '''
      Perform one iteration of EM from ground-truth responsibilities
      Verify that resulting responsibilities don't differ much from the originals
    '''
    LP = dict(resp=self.trueresp)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)
    LP = self.hmodel.calc_local_params(self.Data)
    # respError : N x 1 vector, maximum error in responsibility across all dataitems
    respError = np.max(np.abs(self.trueresp - LP['resp']), axis=1)
    # verify that for most points, the error in resp computation is very low
    assert np.percentile(respError, 50) < 0.005
    assert np.percentile(respError, 90) < 0.05
     
  def test_update_global_params(self):
    '''
      Perform one M-step starting with ground-truth responsibilities
      Verify that the resulting parameters (cov matrix Sigma) don't differ too much
      from the ground truth parameters
    '''
    LP = dict(resp=self.trueresp)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)
    for k in range( self.Sigma.shape[2]):
      curSigma = self.Sigma[:,:,k]
      curSigmaHat = self.hmodel.obsModel.comp[k].ECovMat()
      percDiff = np.abs(curSigma - curSigmaHat)/(0.00001+curSigma)
      absDiff = np.abs(curSigma - curSigmaHat)
      # Each entry in Sigma must be close to the entry in SigmaHat
      #  either in an absolute sense (if true cov is 0, we tolerate answers < eps)
      #  or in a relative sense (measured by percentage difference)
      mask = np.logical_or(percDiff < 0.15, absDiff < 0.05)
      assert np.all(mask)
    
