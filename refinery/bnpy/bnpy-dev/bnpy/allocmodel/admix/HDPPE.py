'''
HDPPE.py
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
v   : K-length vector, point estimate for stickbreaking fractions v1, v2, ... vK
'''
import numpy as np

import OptimizerForHDPPE as OptimPE
from .HDPModel import HDPModel
from ...util import gammaln

class HDPPE(HDPModel):

  ######################################################### Constructors
  #########################################################
  ''' Handled by HDPModel
  '''
        
  def set_helper_params(self):
    self.Ebeta = OptimPE.v2beta(self.v)

  ######################################################### Accessors
  #########################################################
  def get_keys_for_memoized_local_params(self):
    ''' Return list of string names of the LP fields
         that moVB needs to memoize across visits to a particular batch
    '''
    return ['alphaPi']

  ######################################################### Local Params
  #########################################################
  ''' Handled by HDPModel
  '''

  ######################################################### Suff Stats
  #########################################################
  ''' Handled by HDPModel
  '''

  ######################################################### Global Params
  #########################################################
  def update_global_params_VB(self, SS, **kwargs):
    ''' Update global parameters v that control topic probabilities beta
    '''
    self.K = SS.K
    sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
    assert sumLogPi.size == self.K + 1
    
    try:
      v, fofv, Info = OptimPE.estimate_v(alpha0=self.alpha0, gamma=self.gamma,
                            sumLogPi=sumLogPi, nDoc=SS.nDoc)
      self.v = v
    except ValueError as v:
      if str(v).count('failed') > 0:
        if self.v.size != self.K:
          beta = np.hstack([SS.N, 0.01])
          beta /= beta.sum()
          self.v = OptimPE.beta2v(beta)
        else:
          pass # keep current estimate of v!
    assert self.v.size == self.K
    self.set_helper_params()
          
  
  def update_global_params_soVB(self, SS, rho, **kwargs):
    '''
    '''
    raise NotImplementedError("TODO")

  def set_global_params(self, K=0, beta=None, U1=None, U0=None, 
                                Ebeta=None, EbetaLeftover=None, **kwargs):
    ''' Directly set global parameter vector v
          using provided arguments
    '''
    self.K = K
    if U1 is not None and U0 is not None:
      self.v = U1 / (U1 + U0)
    
    if Ebeta is not None and EbetaLeftover is not None:
      Ebeta = np.squeeze(Ebeta)
      EbetaLeftover = np.squeeze(EbetaLeftover)
      beta = np.hstack( [Ebeta, EbetaLeftover])
    elif beta is not None:
      assert beta.size == K
      beta = np.hstack([beta, np.min(beta)/100.])
      beta = beta/np.sum(beta)
    if beta is not None:
      assert abs(np.sum(beta) - 1.0) < 0.005
      self.v = OptimPE.beta2v(beta)
    assert self.v.size == self.K
    self.set_helper_params()
        
  ######################################################### Evidence
  #########################################################  
  ''' calc_evidence inherited from HDPModel
  '''

  ####################################################### ELBO terms for Z
  ''' inherited from HDPModel
  '''
  
  ####################################################### ELBO terms for Pi
  def E_logpPi(self, SS):
    ''' Returns scalar value of E[ log p(PI | alpha0)]
    '''
    K = SS.K
    kvec = K + 1 - np.arange(1, K+1)
    # logDirNormC : scalar norm const that applies to each iid draw pi_d
    logDirNormC = gammaln(self.gamma) - np.sum(gammaln(self.gamma*self.Ebeta))
    # logDirPDF : scalar sum over all doc's pi_d
    sumLogPi = np.hstack([SS.sumLogPiActive, SS.sumLogPiUnused])
    logDirPDF = np.inner(self.gamma * self.Ebeta - 1., sumLogPi)
    return (SS.nDoc * logDirNormC) + logDirPDF


  ####################################################### ELBO terms for V
  def E_logpV(self):
    logBetaNormC = gammaln(self.alpha0 + 1.) - gammaln(self.alpha0)
    logBetaPDF = (self.alpha0-1.) * np.sum(np.log(1-self.v))
    return self.K*logBetaNormC + logBetaPDF

  def E_logqV(self):
    ''' Returns entropy of q(v), which for a point estimate is always 0
    '''
    return 0

  ####################################################### ELBO terms merge
  ''' Inherited from HDPModel
  '''

  ######################################################### IO Utils
  #########################################################   for humans
  def get_info_string( self):
    ''' Returns human-readable name of this object
    '''
    s = 'HDP model. K=%d, alpha=%.2f, gamma=%.2f. Point estimates v.'
    return s % (self.K, self.alpha0, self.gamma)
     
  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict( self ):
    return dict(v=self.v)              
  
  def from_dict(self, Dict):
    self.inferType = Dict['inferType']
    self.v = np.squeeze(np.asarray(Dict['v'], dtype=np.float64))
    self.K = self.v.size
    self.set_helper_params()

  def get_prior_dict( self ):
    return dict(K=self.K, alpha0=self.alpha0, gamma=self.gamma)

