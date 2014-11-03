'''
Unit tests for applying MergeMove.py to HDPModel.py

Verification of math behind merge evidence bound (ELBO) calculation.

Attributes
-----------
self.Data : K=4 simple WordsData object from AbstractBaseTestForHDP
self.hmodel : K=4 simple bnpy model from AbstractBaseTestForHDP

Coverage
-----------
* ELBO Z terms
* ELBO Pi terms
* entire ELBO

'''

import numpy as np
import unittest

import bnpy
from bnpy.learnalg import MergeMove
from scipy.special import digamma
import copy

from AbstractBaseTestForHDP import AbstractBaseTestForHDP

class TestMathForHDPMerges(AbstractBaseTestForHDP):

  ######################################################### Direct construct 
  #########################################################  LP for merge kA,kB
  def calc_mergeLP(self, LP, kA, kB):
    ''' Calculate and return the new local parameters for a merged configuration
          that combines topics kA and kB

        Returns
        ---------
        LP : dict of local params, with updated fields and only K-1 comps
    '''
    K = LP['DocTopicCount'].shape[1]
    assert kA < kB
    LP = copy.deepcopy(LP)
    for key in ['DocTopicCount', 'word_variational', 'alphaPi']:
      LP[key][:,kA] = LP[key][:,kA] + LP[key][:,kB]
      LP[key] = np.delete(LP[key], kB, axis=1)
    LP['E_logPi'] = digamma(LP['alphaPi']) \
                    - digamma(LP['alphaPi'].sum(axis=1))[:,np.newaxis]
    assert LP['word_variational'].shape[1] == K-1
    return LP


  ######################################################### Verify Z terms
  #########################################################  
  def test_ELBO_Z_terms_are_correct_alldata(self):
    ''' Verify that memoized calc of Z terms stored in SS exactly match
          same terms computed directly from local parameters,
          for the true self.hmodel, and the full dataset self.Data

        Relevant terms:  ElogpZ, ElogqZ for assignment variables Z
    '''
    aModel = self.hmodel.allocModel
    LP = self.hmodel.calc_local_params(self.Data)
    # 1) Calculate the ELBO terms directly
    directElogpZ = np.sum(aModel.E_logpZ(self.Data, LP))
    directElogqZ = np.sum(aModel.E_logqZ(self.Data, LP))
    # 2) Calculate the terms via suff stats
    SS = self.hmodel.get_global_suff_stats(self.Data, LP, doPrecompEntropy=True)
    memoElogpZ = np.sum(SS.getELBOTerm('ElogpZ'))
    memoElogqZ = np.sum(SS.getELBOTerm('ElogqZ'))
    assert np.allclose(directElogpZ, memoElogpZ)
    assert np.allclose(directElogqZ, memoElogqZ)

  def test_ELBO_Z_terms_are_correct_minibatches(self):
    ''' Verify that memoized ELBO terms for ElogpZ, ElogqZ stored in SS
          exactly match the same terms computed directly from local params.
          for the true self.hmodel and 2 *minibatches* of the full dataset.        
    '''
    aModel = self.hmodel.allocModel
    LP1 = self.hmodel.calc_local_params(self.batchData1)
    LP2 = self.hmodel.calc_local_params(self.batchData2)
    # 1) Calculate the ELBO terms directly
    directElogpZ = np.sum(aModel.E_logpZ(self.batchData1, LP1) \
                         + aModel.E_logpZ(self.batchData2, LP2)
                         )
    directElogqZ = np.sum(aModel.E_logqZ(self.batchData1, LP1) \
                         + aModel.E_logqZ(self.batchData2, LP2)
                         )
    # 2) Calculate via aggregated suff stats
    SS1 = self.hmodel.get_global_suff_stats(self.batchData1, LP1, doPrecompEntropy=True)
    SS2 = self.hmodel.get_global_suff_stats(self.batchData2, LP2, doPrecompEntropy=True)
    SS = SS1 + SS2
    memoElogpZ = np.sum(SS.getELBOTerm('ElogpZ'))
    memoElogqZ = np.sum(SS.getELBOTerm('ElogqZ'))
    assert np.allclose(directElogpZ, memoElogpZ)
    assert np.allclose(directElogqZ, memoElogqZ)

  def test_ELBO_Z_terms_are_correct_merge(self):
    ''' Verify that the ELBO terms for ElogpZ, ElogqZ are correct
        when a merge occurs
    '''
    aModel = self.hmodel.allocModel
    LP1 = self.hmodel.calc_local_params(self.batchData1)
    LP2 = self.hmodel.calc_local_params(self.batchData2)
 
    beforeElogqZ = aModel.E_logqZ(self.batchData1, LP1) \
                 + aModel.E_logqZ(self.batchData2, LP2)
    print beforeElogqZ
    # ----------------------------------------  Consider merge of comps 0 and 1
    kA = 0
    kB = 1
    mLP1 = self.calc_mergeLP(LP1, kA, kB)
    mLP2 = self.calc_mergeLP(LP2, kA, kB)
    # 1) Calculate the ELBO terms directly
    directElogpZ = aModel.E_logpZ(self.batchData1, mLP1) \
                 + aModel.E_logpZ(self.batchData2, mLP2)
                         
    directElogqZ = aModel.E_logqZ(self.batchData1, mLP1) \
                 + aModel.E_logqZ(self.batchData2, mLP2)
                         
    assert LP1['DocTopicCount'].shape[1] == aModel.K # still have originals!
    assert LP2['DocTopicCount'].shape[1] == aModel.K # still have originals!
    # 2) Calculate via suff stats
    SS1 = self.hmodel.get_global_suff_stats(self.batchData1, LP1,
                           doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS2 = self.hmodel.get_global_suff_stats(self.batchData2, LP2,
                           doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS = SS1 + SS2
    SS.mergeComps(kA, kB)
    memoElogpZ = SS.getELBOTerm('ElogpZ')
    memoElogqZ = SS.getELBOTerm('ElogqZ')
    assert np.allclose(directElogpZ, memoElogpZ)
    assert np.allclose(directElogqZ, memoElogqZ)
    # ----------------------------------------  Follow-up merge of comps 2 and 3
    # since we've collapsed 0,1 into 0 previously,
    #  original comps 2,3 have been "renamed" 1,2
    kA = 1
    kB = 2
    mLP1 = self.calc_mergeLP(mLP1, kA, kB)
    mLP2 = self.calc_mergeLP(mLP2, kA, kB)
    assert mLP1['DocTopicCount'].shape[1] == aModel.K - 2 # we've done two merges
    # 1) Calculate the ELBO terms directly
    directElogpZ = aModel.E_logpZ(self.batchData1, mLP1) \
                 + aModel.E_logpZ(self.batchData2, mLP2)
                         
    directElogqZ = aModel.E_logqZ(self.batchData1, mLP1) \
                 + aModel.E_logqZ(self.batchData2, mLP2)
    # 2) Calculate via suff stats (carried over from earlier merge!)
    SS.mergeComps(kA, kB)
    memoElogpZ = SS.getELBOTerm('ElogpZ')
    memoElogqZ = SS.getELBOTerm('ElogqZ')
    assert np.allclose(directElogpZ, memoElogpZ)
    assert np.allclose(directElogqZ, memoElogqZ)
    # Make sure we don't have any more valid precomputed merge terms
    WMat = SS.getMergeTerm('ElogpZ')
    numZeros = np.sum( WMat == 0)
    numNans = np.isnan(WMat).sum()
    assert numZeros + numNans == WMat.size

  ######################################################### Verify Pi terms
  ######################################################### 

  def test_ELBO_Pi_terms_are_correct(self):
    ''' Verify that the ELBO terms for ElogpPi, ElogqPi are correct
    '''
    aModel = self.hmodel.allocModel
    LP = self.hmodel.calc_local_params(self.Data)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP, doPrecompEntropy=True)
    # 1) Calculate the ELBO terms directly
    directElogpPi = np.sum(aModel.E_logpPi(SS))
    directElogqPi = np.sum(aModel.E_logqPi(LP))
    # 2) Calculate the terms via suff stats
    memoElogpPi = np.sum(aModel.E_logpPi(SS))
    memoElogqPi = np.sum(SS.getELBOTerm('ElogqPiActive')) \
                    + SS.getELBOTerm('ElogqPiConst') \
                    + SS.getELBOTerm('ElogqPiUnused')
    assert np.allclose(directElogpPi, memoElogpPi)
    assert np.allclose(directElogqPi, memoElogqPi)

  def test_ELBO_Pi_terms_are_correct_memoized(self):
    ''' Verify that the ELBO terms for ElogpPi, ElogqPi are correct
    '''
    aModel = self.hmodel.allocModel
    LP1 = self.hmodel.calc_local_params(self.batchData1)
    LP2 = self.hmodel.calc_local_params(self.batchData2)

    SS1 = self.hmodel.get_global_suff_stats(self.batchData1, LP1,   doPrecompEntropy=True)
    SS2 = self.hmodel.get_global_suff_stats(self.batchData2, LP2, doPrecompEntropy=True)
    SS = SS1 + SS2
    assert SS.nDoc == self.Data.nDocTotal

    # 1) Calculate the ELBO terms directly
    directElogpPi = np.sum(aModel.E_logpPi(SS))
    directElogqPi = np.sum(aModel.E_logqPi(LP1) + aModel.E_logqPi(LP2))

    # 2) Calculate the terms via suff stats
    memoElogpPi = np.sum(aModel.E_logpPi(SS))
    memoElogqPi = np.sum(SS.getELBOTerm('ElogqPiActive')) \
                    + SS.getELBOTerm('ElogqPiConst') \
                    + SS.getELBOTerm('ElogqPiUnused')

    assert np.allclose(directElogpPi, memoElogpPi)
    assert np.allclose(directElogqPi, memoElogqPi)

  

  def test_ELBO_Pi_terms_are_correct_merge(self):
    ''' Verify that the ELBO terms for ElogpPi, ElogqPi are correct
    '''
    aModel = self.hmodel.allocModel
    LP1 = self.hmodel.calc_local_params(self.batchData1)
    LP2 = self.hmodel.calc_local_params(self.batchData2)

    SS1 = self.hmodel.get_global_suff_stats(self.batchData1, LP1,   doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS2 = self.hmodel.get_global_suff_stats(self.batchData2, LP2, doPrecompEntropy=True, doPrecompMergeEntropy=True)

    SS = SS1 + SS2
    assert SS.nDoc == self.Data.nDocTotal
    cTerm = SS.getELBOTerm('ElogqPiConst')
    cTerm1 = SS1.getELBOTerm('ElogqPiConst')
    cTerm2 = SS2.getELBOTerm('ElogqPiConst')
    assert np.allclose(cTerm, cTerm1 + cTerm2)

    # Consider merge of comps 0 and 1
    K = SS.K
    print "BEFORE sumLogPiActive", SS.sumLogPiActive
    kA = 0
    kB = 1
    mLP1 = self.calc_mergeLP(LP1, kA, kB)
    mLP2 = self.calc_mergeLP(LP2, kA, kB)
    mergeSumLogPi_kA = SS.getMergeTerm('sumLogPiActive')[kA,kB]

    print "EXPECTED new entry: ", SS.getMergeTerm('sumLogPiActive')[kA,kB]

    SS.mergeComps(kA, kB)
    print "AFTER sumLogPiActive", SS.sumLogPiActive

    assert np.allclose(SS.getELBOTerm('ElogqPiConst'), cTerm1 + cTerm2)

    sumLogPi = mLP1['E_logPi'].sum(axis=0) + mLP2['E_logPi'].sum(axis=0)
    sumLogPiVec = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])

    assert sumLogPi[kA] == mergeSumLogPi_kA
    assert sumLogPiVec[kA] == mergeSumLogPi_kA

    print sumLogPi, sumLogPiVec
    assert np.allclose(sumLogPi, sumLogPiVec)

    # 1) Calculate the ELBO terms directly
    directElogqPi = np.sum(aModel.E_logqPi(mLP1) + aModel.E_logqPi(mLP2))

    # 2) Calculate the terms via suff stats
    memoElogqPi = np.sum(SS.getELBOTerm('ElogqPiActive')) \
                    + SS.getELBOTerm('ElogqPiConst') \
                    + SS.getELBOTerm('ElogqPiUnused')
    print directElogqPi
    print memoElogqPi
    assert np.allclose(directElogqPi, memoElogqPi)

    # ----------------------------------------  Follow-up merge of comps 2 and 3
    # since we've collapsed 0,1 into 0 previously,
    #  original comps 2,3 have been "renamed" 1,2
    kA = 1
    kB = 2
    mLP1 = self.calc_mergeLP(mLP1, kA, kB)
    mLP2 = self.calc_mergeLP(mLP2, kA, kB)
    assert mLP1['DocTopicCount'].shape[1] == aModel.K - 2 # we've done two merges
    # 1) Calculate the ELBO terms directly
    directElogqPi = np.sum(aModel.E_logqPi(mLP1) + aModel.E_logqPi(mLP2))
                         
    # 2) Calculate via suff stats (carried over from earlier merge!)
    SS.mergeComps(kA, kB)
    memoElogqPi = np.sum(SS.getELBOTerm('ElogqPiActive')) \
                    + SS.getELBOTerm('ElogqPiConst')\
                    + SS.getELBOTerm('ElogqPiUnused')
    assert np.allclose(directElogqPi, memoElogqPi)



  ######################################################### Verify full ELBO
  ######################################################### 

  def test_ELBO_terms_are_correct_merge_true(self):
    ''' Verify that entire ELBO is correct
    '''
    np.set_printoptions(precision=4, suppress=True)
    kA = 2
    kB = 3

    LP = self.hmodel.calc_local_params(self.Data)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP,   doPrecompEntropy=True, doPrecompMergeEntropy=True)
    beforeELBO = self.hmodel.calc_evidence(self.Data, SS, LP)

    # Perform merge via suff stats
    SS.mergeComps(kA,kB)
    newModel = self.hmodel.copy()
    newModel.update_global_params(SS)
    memoELBO = newModel.calc_evidence(SS=SS)
    assert newModel.allocModel.K == 3

    # Perform merge via direct operation on true components
    assert self.hmodel.allocModel.K == 4
    mLP = self.calc_mergeLP(LP, kA, kB)
    mSS = self.hmodel.get_global_suff_stats(self.Data, mLP)
    assert np.allclose(mSS.WordCounts, SS.WordCounts)
    assert np.allclose(mSS.sumLogPiActive, SS.sumLogPiActive)

    self.hmodel.update_global_params(mSS)
    assert self.hmodel.allocModel.K == 3
    directELBO = self.hmodel.calc_evidence(self.Data, mSS, mLP)

    print beforeELBO
    print directELBO
    print memoELBO
    assert np.allclose(directELBO, memoELBO)
    assert beforeELBO > directELBO


  def test_ELBO_terms_are_correct_merge_duplicates(self):
    ''' Verify that the ELBO terms for ElogpPi, ElogqPi are correct
    '''
    np.set_printoptions(precision=4, suppress=True)
    kA = 2
    kB = 3
    self.MakeModelWithDuplicatedComps()

    LP = self.dupModel.calc_local_params(self.Data)
    SS = self.dupModel.get_global_suff_stats(self.Data, LP,   doPrecompEntropy=True, doPrecompMergeEntropy=True)
    beforeELBO = self.dupModel.calc_evidence(self.Data, SS, LP)

    # Perform merge via suff stats
    SS.mergeComps(kA,kB)
    newModel = self.dupModel.copy()
    newModel.update_global_params(SS)
    memoELBO = newModel.calc_evidence(SS=SS)
    assert newModel.allocModel.K == 7

    # Perform merge via direct operation on true components
    assert self.dupModel.allocModel.K == 8
    mLP = self.calc_mergeLP(LP, kA, kB)
    mSS = self.dupModel.get_global_suff_stats(self.Data, mLP)
    assert np.allclose(mSS.WordCounts, SS.WordCounts)
    assert np.allclose(mSS.sumLogPiActive, SS.sumLogPiActive)

    self.dupModel.update_global_params(mSS)
    assert self.dupModel.allocModel.K == 7
    directELBO = self.dupModel.calc_evidence(self.Data, mSS, mLP)


    print beforeELBO
    print directELBO
    print memoELBO
    assert np.allclose(directELBO, memoELBO)
    assert beforeELBO > directELBO
