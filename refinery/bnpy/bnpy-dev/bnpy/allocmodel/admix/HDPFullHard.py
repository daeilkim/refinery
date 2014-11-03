'''
HDPFullHard.py
Bayesian nonparametric admixture model with unbounded number of components K,
  using hard assignments for discrete variable Z,
    and full posterior for global stick-breaking weights v
'''
import numpy as np

from .HDPModel import HDPModel
from bnpy.suffstats import SuffStatBag

import scipy.sparse
import logging
Log = logging.getLogger('bnpy')

class HDPFullHard(HDPModel):
    
    def get_word_variational( self, Data, LP):
        ''' Update and return word-topic assignment variational parameters
        '''
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
        # Take exp of wv in numerically stable manner (first subtract the max)
        #  in-place so no new allocations occur
        colIDs = np.argmax(wv, axis=1)
        # TODO: worry about sparsity of hard assign mat?
        R = scipy.sparse.csr_matrix(
                  (np.ones(Data.nObs), colIDs, np.arange(Data.nObs+1)),
                  shape=(Data.nObs, K), dtype=np.float64)
        LP['word_variational'] = R.toarray()
        assert np.allclose(1.0, np.sum(LP['word_variational'], axis=1))
        return LP

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
            # Z terms
            SS.setELBOTerm('ElogpZ', self.E_logpZ(Data, LP), dims='K')
            # Pi terms
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

        elbo = E_logpPi - E_logqPi \
               + E_logpZ  \
               + E_logpV - E_logqV
        return elbo

