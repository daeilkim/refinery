'''
HDPModelLP2.py
Bayesian nonparametric admixture model with unbounded number of components K

Attributes
-------
K        : # of components
alpha0   : scalar concentration param for global-level stick-breaking params v
gamma    : scalar conc. param for document-level mixture weights pi[d]

Local Parameters (document-specific)
--------
alphaPi : nDoc x K matrix, 
             row d has params for doc d's distribution pi[d] over the K topics
             q( pi[d] ) ~ Dir( alphaPi[d] )
E_logPi : nDoc x K matrix
             row d has E[ log pi[d] ]
DocTopicCount : nDoc x K matrix
                  entry d,k gives the expected number of times
                              that topic k is used in document d
word_variational : nDistinctWords x K matrix
                  row i has params for word i's Discrete distr over K topics

Global Parameters (shared across all documents)
--------
U1, U0   : K-length vectors, params for variational distribution over 
           stickbreaking fractions v1, v2, ... vK
            q(v[k]) ~ Beta(U1[k], U0[k])
'''
import numpy as np

import OptimizerForHDPFullVarModel as OptimHDP
from .HDPModel import HDPModel

from bnpy.suffstats import SuffStatBag
from ...util import NumericUtil
from ...util import digamma, gammaln
from ...util import EPS, np2flatstr
import logging
Log = logging.getLogger('bnpy')


class HDPModelLP2(HDPModel):

  ######################################################### Local Params
  #########################################################

  def calc_local_params(self, Data, LP, 
                          nCoordAscentItersLP=20, 
                          convThrLP=0.01, doOnlySomeDocsLP=False, **kwargs):
    ''' Calculate document-specific quantities (E-step)
          Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
            (2) Approx posterior on doc-topic probabilities

          Returns
          -------
          LP : local params dict, with fields
              alphaPi : nDoc x K+1 matrix, 
                 row d has params for doc d's Dirichlet over K+1 topics
              E_logPi : nDoc x K+1 matrix,
                 row d has doc d's expected log probability of each topic
              word_variational : nDistinctWords x K matrix
                 row i has params for word i's Discrete distr over K topics
              DocTopicCount : nDoc x K matrix
    '''
    # When given no local params LP as input, need to initialize from scratch
    # this forces likelihood to drive the first round of local assignments
    if 'alphaPi' not in LP:
      LP['alphaPi'] = np.ones((Data.nDoc,self.K+1))
    else:
      assert LP['alphaPi'].shape[1] == self.K + 1

    # Allocate other doc-specific variables
    LP['DocTopicCount'] = np.zeros((Data.nDoc, self.K))
    LP['E_logPi'] = digamma(LP['alphaPi']) \
                        - digamma(np.sum(LP['alphaPi'], axis=1))[:,np.newaxis]

    # Precompute ONCE exp( E_logsoftevidence )
    expEloglik = LP['E_logsoftev_WordsData']
    expEloglik -= expEloglik.max(axis=1)[:,np.newaxis] 
    np.exp(expEloglik, out=expEloglik)

    # rSum : nDistinctWords-length vector, row n = \sum_{k} r_{nk}, 
    #         where r_nk is responsibility of topic k for word token n
    rSum = np.zeros(Data.nObs)

    # Repeat until old_alphaPi has stopped changing...
    old_alphaPi = LP['alphaPi'].copy()

    docIDs = range(Data.nDoc)

    for ii in xrange(nCoordAscentItersLP):

      # nDoc x K matrix
      if len(docIDs) == Data.nDoc:
        expElogpi = np.exp(LP['E_logPi'][:, :-1])    
      else:
        expElogpi[docIDs] = np.exp(LP['E_logPi'][docIDs, :-1])    

      for d in docIDs:
        start = Data.doc_range[d,0]
        stop  = Data.doc_range[d,1]
        # subset of expEloglik, has rows belonging to doc d 
        expEloglik_d = expEloglik[start:stop]
        
        # rSum_d : subset of rSum vector belonging to doc d
        rSum_d = np.dot(expEloglik_d, expElogpi[d])
        rSum[start:stop] = rSum_d

        # Update DocTopicCount field of LP
        LP['DocTopicCount'][d,:] = np.dot(expEloglik_d.T, 
                                    Data.word_count[start:stop] / rSum_d
                                         )

      # Element-wise multiply with nDoc x K prior prob matrix
      LP['DocTopicCount'][docIDs] *= expElogpi[docIDs]

      # Update doc_variational field of LP
      LP['alphaPi'] = np.tile(self.gamma * self.Ebeta, (Data.nDoc,1))
      LP['alphaPi'][:,:-1] += LP['DocTopicCount']

      # Update expected value of log(Pi)
      digamma(LP['alphaPi'], out=LP['E_logPi'])
      LP['E_logPi'] -= digamma(np.sum(LP['alphaPi'],axis=1))[:,np.newaxis]

      # Assess convergence
      docDiffs = np.max(np.abs(old_alphaPi - LP['alphaPi']), axis=1)
      if np.max(docDiffs) < convThrLP:
        break
      if doOnlySomeDocsLP:
        docIDs = np.flatnonzero(docDiffs > convThrLP)

      # Store previous value for next convergence test
      # the "[:]" syntax ensures we do NOT copy data over
      old_alphaPi[:] = LP['alphaPi']

    LP['docIDs'] = docIDs
    LP['word_variational'] = expEloglik
    for d in xrange(Data.nDoc):
      start = Data.doc_range[d,0]
      stop  = Data.doc_range[d,1]
      LP['word_variational'][start:stop] *= expElogpi[d]
    LP['word_variational'] /= rSum[:, np.newaxis]

    del LP['E_logsoftev_WordsData']
    assert np.allclose( LP['word_variational'].sum(axis=1), 1.0)
    return LP


  def orig_local_params(self, Data, LP, nCoordAscentItersLP=20, convThrLP=0.01, doDocTopicFracLP=0, **kwargs):
    ''' Calculate document-specific quantities (E-step)
        Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
                 q(word_variational | word_token_variables)
            (2) Approx posterior on doc-topic probabilities
                 q(doc_variational | document_topic_variables)

        Returns
        -------
        LP : local params dict, with fields
              Pi : nDoc x K+1 matrix, 
                 row d has params for doc d's Dirichlet over K+1 topics
              word_variational : nDistinctWords x K matrix
                 row i has params for word i's Discrete distr over K topics
              DocTopicCount : nDoc x K matrix
    '''
    # When given no prev. local params LP, need to initialize from scratch
    # this forces likelihood to drive the first round of local assignments
    if 'alphaPi' not in LP:
      LP['alphaPi'] = np.ones((Data.nDoc,self.K+1))
    else:
      assert LP['alphaPi'].shape[1] == self.K + 1

    LP = self.calc_ElogPi(LP)
    alphaPi_old = LP['alphaPi']
        
    # Allocate lots of memory once
    if 'word_variational' in LP:
      del LP['word_variational']
    LP['word_variational'] = np.zeros(LP['E_logsoftev_WordsData'].shape)
    LP['DocTopicCount'] = np.zeros((Data.nDoc,self.K))

    # Repeat until converged...
    for ii in xrange(nCoordAscentItersLP):
      # Update word_variational field of LP
      LP = self.get_word_variational(Data, LP)
        
      # Update DocTopicCount field of LP
      for d in xrange(Data.nDoc):
        start,stop = Data.doc_range[d,:]
        LP['DocTopicCount'][d,:] = np.dot(
                                     Data.word_count[start:stop],        
                                     LP['word_variational'][start:stop,:]
                                         )
      # Update doc_variational field of LP
      LP = self.get_doc_variational(Data, LP)
      LP = self.calc_ElogPi(LP)

      # Assess convergence
      assert id(alphaPi_old) != id(LP['alphaPi'])
      if np.allclose(alphaPi_old, LP['alphaPi'], atol=convThrLP):
        break
      alphaPi_old = LP['alphaPi']

    if doDocTopicFracLP:
      LP['DocTopicFrac'] = LP['DocTopicCount'].copy()
      LP['DocTopicFrac'] /= LP['DocTopicFrac'].sum(axis=1)[:,np.newaxis]
    return LP


  def forloop_local_params(self, Data, LP, 
                          nCoordAscentItersLP=20, 
                          convThrLP=0.01, **kwargs):
        ''' Calculate document-specific quantities (E-step)
          Alternate updates to two terms until convergence
            (1) Approx posterior on topic-token assignment
                 q(word_variational | word_token_variables)
            (2) Approx posterior on doc-topic probabilities
                 q(doc_variational | document_topic_variables)

          Returns
          -------
          LP : local params dict, with fields
              Pi : nDoc x K+1 matrix, 
                 row d has params for doc d's Dirichlet over K+1 topics
              word_variational : nDistinctWords x K matrix
                 row i has params for word i's Discrete distr over K topics
              DocTopicCount : nDoc x K matrix
        '''
        # When given no prev. local params LP, need to initialize from scratch
        # this forces likelihood to drive the first round of local assignments
        if 'alphaPi' not in LP:
            LP['alphaPi'] = np.ones((Data.nDoc,self.K+1))
        else:
            assert LP['alphaPi'].shape[1] == self.K + 1

        # Assign space
        LP['DocTopicCount'] = np.zeros( (Data.nDoc, self.K))
        LP['E_logPi'] = digamma(LP['alphaPi']) \
                         - digamma( np.sum(LP['alphaPi'], axis=1))[:,np.newaxis]

        expEloglik = LP['E_logsoftev_WordsData']
        expEloglik -= expEloglik.max(axis=1)[:,np.newaxis] 
        np.exp( expEloglik, out=expEloglik)
        LP['word_variational'] = expEloglik
        
        for d in xrange(Data.nDoc):
          start,stop = Data.doc_range[d,:]

          # Nd x K matrix
          expEloglik_d = expEloglik[start:stop]

          # Repeat until old_alphaPi has stopped changing...
          old_alphaPi = LP['alphaPi'][d,:].copy()
          for ii in xrange(nCoordAscentItersLP):
            # K-len vector
            expElogpi_d = np.exp( LP['E_logPi'][d, :-1] )    
    
            # Nd-len vector = product( NdxK matrix, K vector)
            rSum_d = np.dot(expEloglik_d, expElogpi_d)

            # Update DocTopicCount field of LP
            LP['DocTopicCount'][d,:] = np.dot(expEloglik_d.T, 
                                        Data.word_count[start:stop] / rSum_d
                                             )
            LP['DocTopicCount'][d,:] *= expElogpi_d

            # Update doc_variational field of LP
            LP['alphaPi'][d,:] = self.gamma * self.Ebeta
            LP['alphaPi'][d,:-1] += LP['DocTopicCount'][d,:]

            LP['E_logPi'][d,:] = digamma(LP['alphaPi'][d,:]) \
                                  - digamma(np.sum(LP['alphaPi'][d,:]))

            if np.max( np.abs( old_alphaPi - LP['alphaPi'][d,:])) < convThrLP:
              break
            # Store previous value for next convergence test
            # the "[:]" syntax ensures we do NOT copy data over
            old_alphaPi[:] = LP['alphaPi'][d,:]

          LP['word_variational'][start:stop,:] *= expElogpi_d
          LP['word_variational'][start:stop,:] /= rSum_d[:, np.newaxis]
        del LP['E_logsoftev_WordsData']
        return LP