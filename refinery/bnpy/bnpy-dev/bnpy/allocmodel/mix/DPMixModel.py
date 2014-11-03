'''
DPMixModel.py
Bayesian parametric mixture model with a unbounded number of components K

Attributes
-------
  K        : # of components
  alpha0   : scalar concentration hyperparameter of Dirichlet process prior
  
  qalpha0 : K-length vector, params for variational factor q(v)
  qalpha1 : K-length vector, params for variational factor q(v)
            q(v[k]) ~ Beta(qalpha1[k], qalpha0[k])

  truncType : str type of truncation for the Dirichlet Process
              'z' : truncate on the assignments [default]
           or 'v' : truncate stick-breaking distribution
'''
import numpy as np

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericUtil
from bnpy.util import gammaln, digamma, EPS

class DPMixModel(AllocModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, priorDict=None):
    if inferType == 'EM':
      raise ValueError('EM not supported for DPMixModel')
    self.inferType = inferType
    if priorDict is None:
      self.alpha0 = 1.0 # Uniform!
      self.alpha1 = 1.0
      self.truncType = 'z'
    else:
      self.set_prior(priorDict)
    self.K = 0

  def set_prior(self, PriorParamDict):
    self.alpha1 = 1.0
    self.alpha0 = PriorParamDict['alpha0']
    self.truncType = PriorParamDict['truncType']
    
  def set_helper_params( self ):
    ''' Set dependent attributes given primary global params.
        For DP mixture, this means precomputing digammas.
    '''
    DENOM = digamma(self.qalpha0 + self.qalpha1)
    self.ElogV      = digamma(self.qalpha1) - DENOM
    self.Elog1mV    = digamma(self.qalpha0) - DENOM

    if self.truncType == 'v':
      self.qalpha1[-1] = 1
      self.qalpha0[-1] = EPS # avoid digamma(0), which is way too HUGE
      self.ElogV[-1] = 0  # log(1) => 0
      self.Elog1mV[-1] = np.log(1e-40) # log(0) => -INF, never used
		
		# Calculate expected mixture weights E[ log w_k ]	 
    self.Elogw = self.ElogV.copy() #copy so we can do += without modifying ElogV
    self.Elogw[1:] += self.Elog1mV[:-1].cumsum()
    

  ######################################################### Accessors
  #########################################################
  def get_keys_for_memoized_local_params(self):
    ''' Return list of string names of the LP fields
        that this object needs to memoize across visits to a particular batch
    '''
    return list()

  ######################################################### Local Params
  #########################################################
  def calc_local_params(self, Data, LP, **kwargs):
    ''' Calculate local parameters for each data item and each component.    
        This is part of the E-step.
        
        Args
        -------
        Data : bnpy data object with Data.nObs observations
        LP : local param dict with fields
              E_log_soft_ev : Data.nObs x K array
                  E_log_soft_ev[n,k] = log p(data obs n | comp k)
        
        Returns
        -------
        LP : local param dict with fields
              resp : Data.nObs x K array whose rows sum to one
              resp[n,k] = posterior responsibility that comp. k has for data n                
    '''
    lpr = LP['E_log_soft_ev']
    lpr += self.Elogw
    # Calculate exp in numerically stable manner (first subtract the max)
    #  perform this in-place so no new allocations occur
    NumericUtil.inplaceExpAndNormalizeRows(lpr)
    LP['resp'] = lpr
    assert np.allclose(lpr.sum(axis=1), 1)
    return LP

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, LP,
                             doPrecompEntropy=False, 
                             doPrecompMergeEntropy=False, mPairIDs=None):
    ''' Calculate the sufficient statistics for global parameter updates
        Only adds stats relevant for this allocModel. 
        Other stats are added by the obsModel.
        
        Args
        -------
        Data : bnpy data object
        LP : local param dict with fields
              resp : Data.nObs x K array,
                       where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
                      indicates whether to precompute ELBO terms in advance
                      used for memoized learning algorithms (moVB)
        doPrecompMergeEntropy : boolean flag
                      indicates whether to precompute ELBO terms in advance
                      for all possible merges of pairs of components
                      used for optional merge moves

        Returns
        -------
        SS : SuffStats for K components, with field
              N : vector of length-K,
                   effective number of observations assigned to each comp
    '''
    Nvec = np.sum(LP['resp'], axis=0)
    SS = SuffStatBag(K=Nvec.size, D=Data.dim)
    SS.setField('N', Nvec, dims=('K'))
    if doPrecompEntropy:
      ElogqZ_vec = self.E_logqZ(LP)
      SS.setELBOTerm('ElogqZ', ElogqZ_vec, dims=('K'))
    if doPrecompMergeEntropy:
      # Hmerge : KxK matrix of entropies for all possible pair-wise merges
      # for example, if we had only 3 components {0,1,2}
      # Hmerge = [ 0 H(0,1) H(0,2)
      #            0   0    H(1,2)
      #            0   0      0 ]      
      #  where H(i,j) is entropy if components i and j merged.
      Hmerge = np.zeros((self.K, self.K))
      for jj in range(self.K):
        compIDs = np.arange(jj+1, self.K)
        Rcombo = LP['resp'][:,jj][:,np.newaxis] + LP['resp'][:,compIDs]
        Hmerge[jj,compIDs] = np.sum(Rcombo*np.log(Rcombo+EPS), axis=0)
      SS.setMergeTerm('ElogqZ', Hmerge, dims=('K','K'))
    return SS

  ######################################################### Global Params
  #########################################################
  def update_global_params_VB( self, SS, **kwargs ):
    ''' Updates global params (stick-breaking Beta params qalpha1, qalpha0)
          for conventional VB learning algorithm.
        New parameters have exactly the number of components specified by SS. 
    '''
    self.K = SS.K
    qalpha1 = self.alpha1 + SS.N
    qalpha0 = self.alpha0 * np.ones(self.K)
    qalpha0[:-1] += SS.N[::-1].cumsum()[::-1][1:]
    self.qalpha1 = qalpha1
    self.qalpha0 = qalpha0
    self.set_helper_params()
    
  def update_global_params_soVB( self, SS, rho, **kwargs ):
    ''' Update global params (stick-breaking Beta params qalpha1, qalpha0).
        for stochastic online VB.
    '''
    assert self.K == SS.K
    qalpha1 = self.alpha1 + SS.N
    qalpha0 = self.alpha0 * np.ones( self.K )
    qalpha0[:-1] += SS.N[::-1].cumsum()[::-1][1:]
    
    self.qalpha1 = rho * qalpha1 + (1-rho) * self.qalpha1
    self.qalpha0 = rho * qalpha0 + (1-rho) * self.qalpha0
    self.set_helper_params()

  def set_global_params(self, hmodel=None, K=None, qalpha1=None, 
                              qalpha0=None, **kwargs):
    ''' Directly set global parameters qalpha0, qalpha1 to provided values
    '''
    if hmodel is not None:
      self.K = hmodel.allocModel.K
      self.qalpha1 = hmodel.allocModel.qalpha1
      self.qalpha0 = hmodel.allocModel.qalpha0
      self.set_helper_params()
      return
    if type(qalpha1) != np.ndarray or qalpha1.size != K or qalpha0.size != K:
      raise ValueError("Bad DP Parameters")
    self.K = K
    self.qalpha1 = qalpha1
    self.qalpha0 = qalpha0
    self.set_helper_params()
 
  ######################################################### Evidence
  #########################################################
  def calc_evidence(self, Data, SS, LP=None ):
    '''
    '''
    evV = self.E_logpV() - self.E_logqV()
    if SS.hasELBOTerm('ElogqZ'):
      evZq = np.sum(SS.getELBOTerm('ElogqZ'))     
    else:
      evZq = np.sum(self.E_logqZ(LP))
    if SS.hasAmpFactor():
      evZ = self.E_logpZ(SS) -  SS.ampF * evZq
    else:
      evZ = self.E_logpZ(SS) - evZq
    return evZ + evV
         
  def E_logpZ(self, SS):
    return np.inner( SS.N, self.Elogw ) 
    
  def E_logqZ(self, LP):
    return np.sum(LP['resp'] * np.log(LP['resp']+EPS), axis=0)
    
  def E_logpV( self ):
    logNormC = gammaln(self.alpha0 + self.alpha1) \
                    - gammaln(self.alpha0) - gammaln(self.alpha1)
    logBetaPDF = (self.alpha1-1)*self.ElogV + (self.alpha0-1)*self.Elog1mV
    if self.truncType == 'z':
	    return self.K*logNormC + logBetaPDF.sum()    
    elif self.truncType == 'v':
      return self.K*logNormC + logBetaPDF[:-1].sum()

  def E_logqV( self ):
    logNormC = gammaln(self.qalpha0 + self.qalpha1) \
                      - gammaln(self.qalpha0) - gammaln(self.qalpha1)
    logBetaPDF = (self.qalpha1-1)*self.ElogV + (self.qalpha0-1)*self.Elog1mV
    if self.truncType == 'z':
      return logNormC.sum() + logBetaPDF.sum()
    elif self.truncType == 'v':
      # skip last entry because entropy of Beta(1,0) = 0
      return logNormC[:-1].sum() + logBetaPDF[:-1].sum()
    
  ######################################################### IO Utils
  #########################################################   for humans
  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    msgPattern = 'DP mixture with K=%d. Concentration alpha0= %.2f' 
    return msgPattern % (self.K, self.alpha0)

  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict(self): 
    return dict(qalpha1=self.qalpha1, qalpha0=self.qalpha0)
    
  def from_dict(self, myDict):
    self.inferType = myDict['inferType']
    self.K = myDict['K']
    self.qalpha1 = myDict['qalpha1']
    self.qalpha0 = myDict['qalpha0']
    if self.qalpha0.ndim == 0:
      self.qalpha0 = self.qalpha1[np.newaxis]
    if self.qalpha0.ndim == 0:
      self.qalpha0 = self.qalpha0[np.newaxis]
    self.set_helper_params()
    
  def get_prior_dict(self):
    return dict(alpha1=self.alpha1, alpha0=self.alpha0, K=self.K, 
                  truncType=self.truncType)  
