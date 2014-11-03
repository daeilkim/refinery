'''
Unit tests for MergeMove.py for HDPTopicModels

Verification merging works as expected and produces valid models.

Attributes
------------
self.Data : K=4 simple WordsData object from AbstractBaseTestForHDP
self.hmodel : K=4 simple bnpy model from AbstractBaseTestForHDP

Coverage
-----------

* run_many_merge_moves
  * fails to merge away any true comps
  * successfully merges away all duplicated comps when chosen randomly
  * successfully merges away all duplicated comps when chosen via marglik

* run_merge_move
  * fails to merge away any true comps
  * successfully merges away all duplicated comps when targeted specifically
  * successfully merges away all duplicated comps when chosen randomly
  * successfully merges away all duplicated comps when chosen via marglik
      success rate > 95%

   
'''
import numpy as np
import unittest
from AbstractBaseTestForHDP import AbstractBaseTestForHDP

import bnpy
from bnpy.learnalg import MergeMove
from scipy.special import digamma
import copy

class TestMergeHDP(AbstractBaseTestForHDP):

  def getSuffStatsPrepForMerge(self, hmodel):
    ''' With merge flats ENABLED,
          run Estep, calc suff stats, then do an Mstep
    '''
    LP = hmodel.calc_local_params(self.Data)
    flagDict = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS = hmodel.get_global_suff_stats(self.Data, LP, **flagDict)
    hmodel.update_global_params(SS)
    return LP, SS

  ######################################################### Test many moves
  #########################################################
  def test_run_many_merge_moves_trueModel_random(self):
    LP, SS = self.getSuffStatsPrepForMerge(self.hmodel)
    PRNG = np.random.RandomState(0)
    mergeKwArgs = dict(mergename='random')
    a, b, c, MTracker = MergeMove.run_many_merge_moves(self.hmodel, 
                               self.Data, SS,
                               nMergeTrials=100, randstate=PRNG,
                               **mergeKwArgs)
    assert MTracker.nTrial == SS.K * (SS.K-1)/2
    assert MTracker.nSuccess == 0

  def test_run_many_merge_moves_dupModel_random(self):
    self.MakeModelWithDuplicatedComps()
    LP, SS = self.getSuffStatsPrepForMerge(self.dupModel)
    PRNG = np.random.RandomState(0)
    mergeKwArgs = dict(mergename='random')
    a, b, c, MTracker = MergeMove.run_many_merge_moves(self.dupModel, 
                               self.Data, SS,
                               nMergeTrials=100, randstate=PRNG,
                               **mergeKwArgs)
    assert MTracker.nSuccess == 4
    assert (0,4) in MTracker.acceptedOrigIDs
    assert (1,5) in MTracker.acceptedOrigIDs
    assert (2,6) in MTracker.acceptedOrigIDs
    assert (3,7) in MTracker.acceptedOrigIDs

  def test_run_many_merge_moves_dupModel_marglik(self):
    self.MakeModelWithDuplicatedComps()
    LP, SS = self.getSuffStatsPrepForMerge(self.dupModel)
    PRNG = np.random.RandomState(456)
    mergeKwArgs = dict(mergename='marglik')
    a, b, c, MTracker = MergeMove.run_many_merge_moves(self.dupModel, 
                               self.Data, SS,
                               nMergeTrials=100, randstate=PRNG,
                               **mergeKwArgs)
    for msg in MTracker.InfoLog:
      print msg
    assert MTracker.nSuccess == 4
    assert MTracker.nTrial == 4
    assert (0,4) in MTracker.acceptedOrigIDs
    assert (1,5) in MTracker.acceptedOrigIDs
    assert (2,6) in MTracker.acceptedOrigIDs
    assert (3,7) in MTracker.acceptedOrigIDs


  ######################################################### run_merge_move
  #########################################################  full tests
  def test_model_matches_ground_truth_as_precheck(self):
    ''' Verify HDPmodel is able to learn ground truth parameters
          and maintain stable estimates after several E/M steps
    '''
    np.set_printoptions(precision=3,suppress=True)
    # Advance the model several iterations
    for rr in range(5):
      self.run_Estep_then_Mstep()
    for k in range(self.hmodel.obsModel.K):
      logtopicWordHat = self.hmodel.obsModel.comp[k].Elogphi
      topicWordHat = np.exp(logtopicWordHat)
      diffVec = np.abs(topicWordHat - self.Data.TrueParams['topics'][k])
      print diffVec
      print ' '
      assert np.max(diffVec) < 0.04


  ######################################################### run_merge_move
  #########################################################  full tests
  def test_run_merge_move_on_true_comps_fails(self):
    ''' Should not be able to merge "true" components into one another
        Each is necessary to explain (some) data
    '''
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.hmodel.calc_local_params(self.Data)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    for trial in range(10):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.hmodel, self.Data, SS, mergename='random')
      assert newModel.allocModel.K == self.hmodel.allocModel.K
      assert newModel.obsModel.K == self.hmodel.obsModel.K

  def test_run_merge_move_on_dup_comps_succeeds_with_each_ideal_pair(self):
    ''' Given the duplicated comps model,
          which has a redundant copy of each "true" component,
        We show that deliberately merging each pair does succeed.
        This is "ideal" since we know in advance which merge pair to try
    '''
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    for kA in [0,1,2,3]:
      kB = kA + 4 # Ktrue=4, so kA's best match is kA+4 
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel,
                                         self.Data, SS, kA=kA, kB=kB)
      print MoveInfo['msg']
      assert newModel.allocModel.K == self.dupModel.allocModel.K - 1
      assert newModel.obsModel.K == self.dupModel.obsModel.K - 1
      assert MoveInfo['didAccept'] == 1

  def test_run_merge_move_on_dup_comps_fails_with_nonideal_pairs(self):
    ''' Given the duplicated comps model,
          which has a redundant copy of each "true" component,
        We show that deliberately merging each pair does succeed.
        This is "ideal" since we know in advance which merge pair to try
    '''
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    for Kstep in [1,2,3,5,6,7]:
      for kA in range(8 - Kstep):
        kB = kA + Kstep
        newM, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel,
                                         self.Data, SS, kA=kA, kB=kB)
        print MoveInfo['msg']
        assert MoveInfo['didAccept'] == 0


  def test_run_merge_move_on_dup_comps_succeeds_with_all_ideal_pairs(self):
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    myModel = self.dupModel.copy()
    for kA in [3,2,1,0]: # descend backwards so indexing still works
      kB = kA + 4 # Ktrue=4, so kA's best match is kA+4 
      myModel, SS, newEv, MoveInfo = MergeMove.run_merge_move(myModel,
                                         self.Data, SS, kA=kA, kB=kB)
      print MoveInfo['msg']
      assert MoveInfo['didAccept'] == 1

  def test_run_merge_move_on_dup_comps_succeeds_with_random_choice(self):
    ''' Consider Duplicated Comps model.
        Out of (8 choose 2) = 28 possible pairs, 
        exactly 4 produce sensible merges.
        Verify that over many random trials where kA,kB drawn uniformly,
          we obtain a success rate not too different from 4 / 28 = 0.142857
    '''
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    nTrial = 100
    nSuccess = 0
    PRNG = np.random.RandomState(0)
    for trial in range(nTrial):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel, self.Data, SS, mergename='random', randstate=PRNG)
      if MoveInfo['didAccept']:
        print MoveInfo['msg']
        nSuccess += 1
    assert nSuccess > 0
    rate = float(nSuccess)/float(nTrial)
    print "Expected rate: .1428"
    print "Measured rate: %.3f" % (rate)
    assert rate > 0.1
    assert rate < 0.2

  def test_run_merge_move_on_dup_comps_succeeds_with_marglik_choice(self):
    ''' Consider Duplicated Comps model.
        Use marglik criteria to select candidates kA, kB.
        Verify that the merge accept rate is much higher than at random.
        The accept rate should actually be near perfect!
    '''
    self.MakeModelWithDuplicatedComps()
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP, **mergeFlags)
    nTrial = 100
    nSuccess = 0
    PRNG = np.random.RandomState(0)
    for trial in range(nTrial):
      newModel, newSS, newEv, MoveInfo = MergeMove.run_merge_move(self.dupModel, self.Data, SS, mergename='marglik', randstate=PRNG)
      print MoveInfo['msg']
      if MoveInfo['didAccept']:
        nSuccess += 1
    assert nSuccess > 0
    rate = float(nSuccess)/float(nTrial)
    print "Expected rate: >.95"
    print "Measured rate: %.3f" % (rate)
    assert rate > 0.95
