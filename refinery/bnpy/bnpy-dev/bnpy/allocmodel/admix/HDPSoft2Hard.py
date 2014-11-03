
import numpy as np

from .HDPModel import HDPModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil, NumericHardUtil

import scipy.sparse
import logging
Log = logging.getLogger('bnpy')

class HDPSoft2Hard(HDPModel):

  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, nCoordAscentItersLP=20, convThrLP=0.01, nHardItersLP=1, doOnlySomeDocsLP=True, **kwargs):
    ''' Calculate document-specific quantities (E-step) using hard assignments.

        Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
                 q(word_variational | word_token_variables)
            (2) Approx posterior on doc-topic probabilities
                 q(doc_variational | document_topic_variables)

        Finishes with *hard* assignments!

        Returns
        -------
        LP : local params dict, with fields
            Pi : nDoc x K+1 matrix, 
                  row d has params for doc d's Dirichlet over K+1 topics
            word_variational : nDistinctWords x K matrix
                 row i has params for word i's Discrete distr over K topics
            DocTopicCount : nDoc x K matrix
    '''
    # First, run soft assignments for nCoordAscentIters
    LP = self.calc_local_params_fast(Data, LP, 
                                        nCoordAscentItersLP,
                                        convThrLP,
                                        doOnlySomeDocsLP,
                                     )

    # Next, finish with hard assignments
    for rep in xrange(nHardItersLP):
      LP = self.get_hard_word_variational(Data, LP)
      # Update DocTopicCount field of LP
      for d in xrange(Data.nDoc):
        start = Data.doc_range[d,0]
        stop = Data.doc_range[d,1]
        LP['DocTopicCount'][d,:] = np.dot(
                                     Data.word_count[start:stop],        
                                     LP['word_variational'][start:stop,:]
                                       )
      # Update doc_variational field of LP
      LP = self.get_doc_variational(Data, LP)
      LP = self.calc_ElogPi(LP)

    return LP

  def get_hard_word_variational(self, Data, LP):
    ''' Update and return word-topic assignment variational parameters
    '''
    LP['word_variational'] = NumericHardUtil.toHardAssignmentMatrix(
                                                    LP['word_variational'])
    return LP
    """
    # Operate on wv matrix, which is nDistinctWords x K
    #  has been preallocated for speed (so we can do += later)
    wv = LP['word_variational']         
    K = wv.shape[1]        
    # Fill in entries of wv with log likelihood terms
    wv[:] = LP['E_logsoftev_WordsData']
    # Add doc-specific log prior to doc-specific rows
    ElogPi = LP['E_logPi'][:,:K]
    for d in xrange(Data.nDoc):
      wv[Data.doc_range[d,0]:Data.doc_range[d,1], :] += ElogPi[d,:]
    colIDs = np.argmax(wv, axis=1)
    # TODO: worry about sparsity of hard assign mat?
    R = scipy.sparse.csr_matrix(
              (np.ones(Data.nObs), colIDs, np.arange(Data.nObs+1)),
              shape=(Data.nObs, K), dtype=np.float64)
    LP['word_variational'] = R.toarray()
    assert np.allclose(LP['word_variational'].sum(axis=1), 1)
    return LP
    """

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP, doPrecompEntropy=False, 
                                              doPrecompMergeEntropy=False,
                                              mPairIDs=None):
    ''' Count expected number of times each topic is used across all docs    
    '''
    wv = LP['word_variational']
    _, K = wv.shape
    # Turn dim checking off, since some stats have dim K+1 instead of K
    SS = SuffStatBag(K=K, D=Data.vocab_size)
    SS.setField('nDoc', Data.nDoc, dims=None)
    sumLogPi = np.sum(LP['E_logPi'], axis=0)
    SS.setField('sumLogPiActive', sumLogPi[:K], dims='K')
    SS.setField('sumLogPiUnused', sumLogPi[-1], dims=None)

    if 'DocTopicFrac' in LP:
      Nmajor = LP['DocTopicFrac']
      Nmajor[Nmajor < 0.05] = 0
      SS.setField('Nmajor', np.sum(Nmajor, axis=0), dims='K')
    if doPrecompEntropy:
      # ---------------- Z terms
      SS.setELBOTerm('ElogpZ', self.E_logpZ(Data, LP), dims='K')
      # ---------------- Pi terms
      # Note: no terms needed for ElogpPI
      # SS already has field sumLogPi, which is sufficient for this term
      ElogqPiC, ElogqPiA, ElogqPiU = self.E_logqPi_Memoized_from_LP(LP)
      SS.setELBOTerm('ElogqPiConst', ElogqPiC, dims=None)
      SS.setELBOTerm('ElogqPiActive', ElogqPiA, dims='K')
      SS.setELBOTerm('ElogqPiUnused', ElogqPiU, dims=None)

    if doPrecompMergeEntropy:
      ElogpZMat, sLgPiMat, ElogqPiMat = self.memo_elbo_terms_for_merge(LP)
      SS.setMergeTerm('ElogpZ', ElogpZMat, dims=('K','K'))
      SS.setMergeTerm('ElogqPiActive', ElogqPiMat, dims=('K','K'))
      SS.setMergeTerm('sumLogPiActive', sLgPiMat, dims=('K','K'))
    return SS

        
  ######################################################### Evidence
  #########################################################  
  def calc_evidence( self, Data, SS, LP ):
    ''' Calculate ELBO terms related to allocation model
    '''   
    E_logpV = self.E_logpV()
    E_logqV = self.E_logqV()
    
    E_logpPi = self.E_logpPi(SS)
    if SS.hasELBOTerms():
      E_logqPi = SS.getELBOTerm('ElogqPiConst') \
                  + SS.getELBOTerm('ElogqPiUnused') \
                  + np.sum(SS.getELBOTerm('ElogqPiActive'))
      E_logpZ = np.sum(SS.getELBOTerm('ElogpZ'))
    else:
      E_logqPi = self.E_logqPi(LP)
      E_logpZ = np.sum(self.E_logpZ(Data, LP))

    if SS.hasAmpFactor():
      E_logqPi *= SS.ampF
      E_logpZ *= SS.ampF


    elbo = E_logpPi - E_logqPi
    elbo += E_logpZ
    elbo += E_logpV - E_logqV
    return elbo


