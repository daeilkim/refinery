'''
Unit tests for BirthMove.py

Verifies that births produce valid models with expected new components.

Coverage
---------
* learn_fresh_model
   * reproducible (random seed)
   * produces viable model and SS (consistent size, etc.)
   * 
* run_birth_move
   * finds correct components on simple toy dataset


TODO
---------
These methods need coverage
* subsample_data
* cleanup_fresh_model

need to do topic-models,ZMGauss for run_birth_move
'''
import numpy as np
import unittest

from bnpy.learnalg import BirthMove
from bnpy import HModel
from bnpy.util.RandUtil import mvnrand
from bnpy.data import XData

class TestBirthMove(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    self.MakeData()
    self.MakeModelWithOneComp()
    self.MakeModelWithTrueComps()


  def MakeData(self, K=3, Nperclass=1000):
    ''' Creates simple toy dataset for testing.
        Simple 3 component data with eye covar and distinct, well-sep means
          mu0 = [-10, -10]
          mu1 = [0, 0]
          mu2 = [10, 10]
    '''
    PRNG = np.random.RandomState(8675309)
    # Means:  [-10 -10; 0 0; 10 10]
    Mu = np.zeros((3,2))
    Mu[0] = Mu[0] - 10
    Mu[2] = Mu[2] + 10
    # Covariances: identity
    Sigma = np.eye(2)    
    # Generate data from K components, each with Nperclass examples
    self.TrueResp = np.zeros((K*Nperclass,K))
    Xlist = list()
    for k in range(K):
      Xcur = mvnrand(Mu[k], Sigma, Nperclass, PRNG)    
      Xlist.append(Xcur)
      self.TrueResp[k*Nperclass:(k+1)*Nperclass, k] = 1.0
    X = np.vstack(Xlist)
    self.Data = XData(X=X)
    self.Mu = Mu
    assert np.abs(self.TrueResp.sum() - self.Data.nObs) < 1e-2

  def MakeModelWithOneComp(self):
    ''' Create DPMix-Gauss model of self.Data, with K=1 components.
        Store result as "oneModel" in self,
           with updated suff stats field "oneSS"
    '''
    aDict = dict(alpha0=1.0, truncType='z')
    oDict = dict(kappa=1e-7, dF=1, ECovMat='eye', sF=1e-3)
    self.oneModel = HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss', aDict, oDict, self.Data)
    LP = dict(resp=np.ones((self.Data.nObs,1)))
    SS = self.oneModel.get_global_suff_stats(self.Data, LP)
    assert SS.x.size == self.Data.dim
    assert self.oneModel.obsModel.obsPrior.D == self.Data.dim
    self.oneModel.update_global_params(SS)
    LP = self.oneModel.calc_local_params(self.Data)
    self.oneSS = self.oneModel.get_global_suff_stats(self.Data, LP)
    
  def MakeModelWithTrueComps(self):
    ''' Create DPMix-Gauss model of self.Data, with K=3 *true* components.
        Store result as "hmodel" in self.
    '''
    aDict = dict(alpha0=1.0, truncType='z')
    oDict = dict(kappa=1e-7, dF=1, ECovMat='eye', sF=1e-3)
    self.hmodel = HModel.CreateEntireModel('VB', 'DPMixModel', 'Gauss', aDict, oDict, self.Data)
    LP = dict(resp=self.TrueResp)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)


  def verify_proposed_model(self, propModel, propSS):
    ''' Verify provided model and SS are consistent with each-other
    '''
    assert propModel.allocModel.K == propSS.K
    assert propModel.obsModel.K == propSS.K
    assert len(propModel.obsModel.comp) == propSS.K
    if hasattr(propSS, 'N'):
      assert propSS.N.size == propSS.K
    if hasattr(propSS, 'x'):
      assert propSS.x.shape[0] == propSS.K
    # Check stick-break params
    if hasattr(propSS, 'qalpha0'):
      assert propModel.allocModel.qalpha0.size == propSS.K
      assert propModel.allocModel.qalpha1.size == propSS.K

  #########################################################
  #########################################################

  def test_model_matches_ground_truth_as_precheck(self):
    ''' Before learning can proceed, need to ensure the model
          is able to learn ground truth.
    '''
    for k in range(self.hmodel.obsModel.K):
      muHat = self.hmodel.obsModel.get_mean_for_comp(k)
      print muHat, self.Mu[k]
      assert np.max(np.abs(muHat - self.Mu[k])) < 0.5
    LP = self.hmodel.calc_local_params(self.Data)
    absDiff = np.abs(LP['resp'] - self.TrueResp)
    maxDiff = np.max(absDiff, axis=1)
    assert np.sum( maxDiff < 0.1 ) > 0.5 * self.Data.nObs

  ######################################################### learn_fresh_model
  #########################################################

  def test_learn_fresh_model_produces_new_comps(self):
    ''' Verify that learn_fresh_model produces new components  
    '''
    PRNG = np.random.RandomState(0)
    freshModel = self.oneModel.copy()
    assert type(freshModel) == HModel
    freshSS = BirthMove.learn_fresh_model(freshModel, self.Data, Kfresh=10,
                      freshInitName='randexamples', freshAlgName='VB',
                      nFreshLap=50, randstate=PRNG)
    assert freshSS.K > 1
    assert freshSS.K <= freshModel.obsModel.K
    
  def test_learn_fresh_model_reproducible_random_seed(self):
    ''' Verify that learn_fresh_model produces same components
        when called with same targetData and same randstate
    '''
    freshModel = self.oneModel.copy()
    Nvec = list()
    xvec = list()
    for trial in range(3):
      PRNG = np.random.RandomState(8383)
      freshSS = BirthMove.learn_fresh_model(freshModel, self.Data, Kfresh=10,
                      freshInitName='randexamples', freshAlgName='VB',
                      nFreshLap=50, randstate=PRNG)
      Nvec.append(freshSS.N)
      xvec.append(freshSS.x)
    assert np.all(Nvec[0] == Nvec[1])
    assert np.all(Nvec[0] == Nvec[2])
    assert np.all(xvec[0] == xvec[2])
    
  def test_run_birth_move_produces_viable_model(self):
    ''' Verify that the output of run_birth_move
        can be used to create a valid model that can do all necessary subroutines like calc_local_params, etc.
    '''
    PRNG = np.random.RandomState(12345)
    birthArgs = dict(Kfresh=10, freshInitName='randexamples', 
                      freshAlgName='VB', nFreshLap=50)
    newModel, newSS, MInfo = BirthMove.run_birth_move(self.oneModel, 
                      self.Data, self.oneSS, randstate=PRNG, **birthArgs)
    # Try newModel out with some quick calculations
    assert newModel.obsModel.K > self.oneModel.obsModel.K
    LP = newModel.calc_local_params(self.Data)
    SS = newModel.get_global_suff_stats(self.Data, LP)
    assert SS.K == newModel.obsModel.K

  ######################################################### run_birth_move
  #########################################################

  def test_run_birth_move_can_create_necessary_comps(self):
    ''' Can we start with one comp and recover the 3 true ones?
        Of course, depending on the random init we maybe cannot,
          but we should be able to a significant fraction of the time.
    '''
    birthArgs = dict(Kfresh=5, freshInitName='randexamples', 
                      freshAlgName='VB', nFreshLap=50)
    nSuccess = 0
    nTrial = 10
    for trial in range(nTrial):
      PRNG = np.random.RandomState(trial)
      newModel, newSS, MInfo = BirthMove.run_birth_move(self.oneModel, 
                      self.Data, self.oneSS, randstate=PRNG, **birthArgs)
      print newSS.N
      newSS = self._run_LearnAlg_iterations(newModel, self.Data, nIter=10)
      # after refining, we better have true comps!
      topCompIDs = np.argsort(newSS.N)[::-1]
      foundMatch = np.zeros(3)
      for kk in topCompIDs[:3]:
        estMu = newModel.obsModel.get_mean_for_comp(kk)
        print estMu
        for jj in range(3):
          trueMu = self.Mu[jj,:]
          if np.max(trueMu - estMu) < 0.5:
            foundMatch[jj] = 1
      print ' '
      if np.all(foundMatch):
        nSuccess += 1
    assert nSuccess/float(nTrial) > 0.5

  def _run_LearnAlg_iterations(self, hmodel, Data, nIter=1):
    for ii in range(nIter):
      LP = hmodel.calc_local_params(Data)
      SS = hmodel.get_global_suff_stats(Data, LP)
      hmodel.update_global_params(SS)
      return SS
    
