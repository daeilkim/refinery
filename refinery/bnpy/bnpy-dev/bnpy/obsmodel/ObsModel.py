'''
ObsModel.py

Generic abstract observation model for bnpy.
Contains information for accessing and updating
* data-generating parameters for all K components 
* parameters for the prior distribution on these data-generating parameters
'''
import numpy as np
import scipy.linalg
import os
import copy

class ObsModel( object ):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, obsPrior=None, D=None):
    self.inferType = inferType
    self.D = D
    self.obsPrior = obsPrior

  @classmethod
  def CreateWithPrior( cls, inferType, obsPriorParams, Data):
    pass      

  @classmethod
  def CreateWithAllComps( cls, inferType, obsPriorParams, Data):
    pass

  ######################################################### Accessors
  #########################################################  

  ######################################################### Local Params
  #########################################################  E-step
  def calc_local_params( self, Data, LP=dict(), **kwargs):
    if self.inferType == 'EM':
      LP['E_log_soft_ev'] = self.log_soft_ev_mat( Data )
    elif self.inferType.count('VB') >0:
      LP['E_log_soft_ev'] = self.E_log_soft_ev_mat( Data )
    return LP

  def log_soft_ev_mat( self, Data, Krange=None):
    ''' E-step update,  for EM-type inference
    '''
    if Krange is None:
      Krange = xrange(self.K)
    lpr = np.zeros( (Data.nObs, self.K) )
    for k in Krange:
      lpr[:,k] = self.comp[k].log_pdf( Data )
    return lpr
      
  def E_log_soft_ev_mat( self, Data, Krange=None ):
    ''' E-step update, for VB-type inference
    '''    
    if Krange is None:
      Krange = xrange(self.K)
    lpr = np.zeros( (Data.nObs, self.K) )
    for k in Krange:
      lpr[:,k] = self.comp[k].E_log_pdf( Data )
    return lpr

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self):
    pass

  ######################################################### Global Params
  #########################################################  M-step
  def update_global_params(self, SS, rho=None, mergeCompA=None, mergeCompB=None, **kwargs):
    ''' M-step update of global parameters for each component of this obs model.
        After this update, self will have exactly the number of 
          components specified by SS.K.
        If this number is changed, all components are rewritten from scratch.
        Args
        -------
        SS : sufficient statistics object (bnpy.suffstats.SuffStatDict)
        rho : learning rate for current step of stochastic online VB (soVB)

        Returns
        -------
        None (update happens *in-place*).         
    '''
    # Components of updated model exactly match those of suff stats
    self.K = SS.K
    if len(self.comp) != self.K:
      if mergeCompB is None:
        self.comp = [copy.deepcopy(self.obsPrior) for k in xrange(self.K)]
      else:
        for kk in xrange(mergeCompB, SS.K):
          self.comp[kk] = self.comp[kk+1]
        del self.comp[-1]
    assert len(self.comp) == self.K

    if self.inferType == 'EM':
      self.update_obs_params_EM(SS)
    elif self.inferType.count('VB') > 0:
      if rho is None or rho == 1.0:
        self.update_obs_params_VB(SS, mergeCompA=mergeCompA, **kwargs)
      else:
        self.update_obs_params_soVB(SS, rho)
  
  def update_obs_params_EM(self):
    pass
    
  def update_obs_params_VB(self):
    pass
    
  def update_obs_params_soVB(self):
    pass


  ######################################################### Evidence
  ######################################################### 
  def calc_evidence(self):
    pass 
  
  def calcLogMargLikForComp(self, SS, kA, kB=None, **kwargs):
    ''' Calculate the log marginal likelihood of the data assigned
          to the given component (specified by integer ID).
        Requires Data pre-summarized into sufficient stats for each comp.
        If multiple comp IDs are provided, we combine into a "merged" component.
        
        Args
        -------
        SS : bnpy suff stats object
        kA : integer ID of target component to compute likelihood for
        kB : (optional) integer ID of second component.
             If provided, we merge kA, kB into one component for calculation.
        Returns
        -------
        scalar log probability of data assigned to given component(s)
    '''
    if kB is None:
      postDistr = self.obsPrior.get_post_distr(SS, kA, **kwargs)
    else:
      postDistr = self.obsPrior.get_post_distr(SS, kA, kB, **kwargs)
    return postDistr.get_log_norm_const()


  ######################################################### I/O Utils
  #########################################################   for humans
  def get_name(self):
    pass

  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    pass

  def get_info_string_prior( self):
    ''' Returns one-line human-readable terse description of the prior
    '''
    pass

  ######################################################### I/O Utils
  #########################################################   for machines
  def get_prior_dict( self ):
    pass

  def to_dict_essential(self):
    PDict = dict(name=self.__class__.__name__, inferType=self.inferType)
    if hasattr(self,"K"):
      PDict['K']=self.K
    if hasattr(self,'min_covar'):
      PDict['min_covar'] = self.min_covar
    return PDict

