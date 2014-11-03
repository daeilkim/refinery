'''
Unit tests for basic VB learning for the Mix - Gauss hmodel 
'''
import numpy as np
np.set_printoptions(precision=3)

from bnpy import HModel
from bnpy.data import XData
from bnpy.util import RandUtil

########################################## basic tests for 4-cluster model
class TestMixZMVB_4Class2D(object):
  def setUp(self):
    self.MakeData()
    self.MakeHModel()
  
  def MakeHModel(self):
    aPDict = dict(alpha0=1.0)
    oPDict = dict(dF=1, ECovMat='eye', sF=1.0, kappa=1e-6)
    self.hmodel = HModel.CreateEntireModel('VB', 'MixModel', 'Gauss', aPDict, oPDict, self.Data)
  
  def MakeData(self, N=25000):
    S1 = np.asarray([[100, 0], [0, 0.01]])
    Sigma = np.zeros( (2,2,4))
    Sigma[:,:,0] = S1
    Sigma[:,:,1] = RandUtil.rotateCovMat(S1, theta=np.pi/4)
    Sigma[:,:,2] = RandUtil.rotateCovMat(S1, theta=2*np.pi/4)
    Sigma[:,:,3] = RandUtil.rotateCovMat(S1, theta=3*np.pi/4)
    self.Sigma = Sigma
    np.random.seed(505)
    self.mu = 10 * np.random.rand(4, 2)
    Xlist = list()
    Rlist = list()
    for k in range(Sigma.shape[2]):
      curX = RandUtil.mvnrand(self.mu[k], Sigma[:,:,k], N)
      curresp = np.zeros((N,4))
      curresp[:,k] = 1.0
      Xlist.append(curX)
      Rlist.append(curresp)
    X = np.vstack(Xlist)
    self.Data = XData(X=X)
    self.trueresp = np.vstack(Rlist)
    
  def test_deg_freedom(self):
    assert self.hmodel.obsModel.obsPrior.dF > self.hmodel.obsModel.D  
    
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
      Verify that the resulting parameters (mu, Sigma) don't differ too much
      from the ground truth parameters
    '''
    LP = dict(resp=self.trueresp)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)
    K = self.hmodel.allocModel.K
    for k in range(K):
      curMu = self.mu[k]
      curMuHat = self.hmodel.obsModel.comp[k].m
      absDiff = np.abs(curMu - curMuHat)
      percDiff = absDiff /np.abs(1e-8 + np.maximum(curMu,curMuHat))
      print curMu
      print curMuHat
      assert np.all(np.logical_or(percDiff < 0.15, absDiff < 0.05))

    
      curSigma = self.Sigma[:,:,k]
      curSigmaHat = self.hmodel.obsModel.comp[k].ECovMat()
      percDiff = np.abs(curSigma - curSigmaHat)/(1e-5+curSigma)
      absDiff = np.abs(curSigma - curSigmaHat)
      # Each entry in Sigma must be close to the entry in SigmaHat
      #  either in an absolute sense (if true cov is 0, we tolerate answers < eps)
      #  or in a relative sense (measured by percentage difference)
      mask = np.logical_or(percDiff < 0.15, absDiff < 0.05)
      assert np.all(mask)
    
