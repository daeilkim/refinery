'''
ZMGaussObsModel.py

Multivariate, zero-mean Gaussian observation model for bnpy.

Contains information for accessing and updating
* data-generating parameters for all K components 
* parameters for the prior distribution on these data-generating parameters
'''
import numpy as np
import scipy.linalg
import os
import copy

from ObsModel import ObsModel
from bnpy.distr import ZMGaussDistr
from bnpy.distr import WishartDistr
from bnpy.util import np2flatstr, dotATA, dotATB, dotABT
from bnpy.util import LOGPI, LOGTWOPI, LOGTWO, EPS

class ZMGaussObsModel(ObsModel):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, D=None, obsPrior=None, min_covar=None):
    self.inferType = inferType
    self.D = D
    self.obsPrior = obsPrior
    self.comp = list()
    self.K = 0
    if min_covar is not None:
      self.min_covar = min_covar

  @classmethod
  def CreateWithPrior(cls, inferType, priorArgDict, Data):
    D = Data.dim
    if inferType == 'EM':
      obsPrior = None
      return cls(inferType, D, obsPrior, min_covar=priorArgDict['min_covar'])
    else:
      obsPrior = WishartDistr.CreateAsPrior(priorArgDict, Data)
      return cls(inferType, D, obsPrior)

  @classmethod
  def CreateWithAllComps(cls, oDict, obsPrior, compDictList):
    if 'min_covar' in oDict:
      mc = oDict['min_covar']
      self = cls(oDict['inferType'], obsPrior=obsPrior, min_covar=mc)
    else:
      self = cls(oDict['inferType'], obsPrior=obsPrior)
    self.K = len(compDictList)
    self.comp = [None for k in range(self.K)]
    if self.inferType == 'EM':
      for k in xrange(self.K):
        self.comp[k] = ZMGaussDistr( **compDictList[k] )
        self.D = self.comp[k].D
    elif self.inferType.count('VB') > 0:
      for k in xrange(self.K):
        self.comp[k] = WishartDistr( **compDictList[k] )
        self.D = self.comp[k].D
    return self

  ######################################################### Accessors
  #########################################################  
  def get_mean_for_comp( self, k):
    return np.zeros( self.D )
    
  def get_covar_mat_for_comp(self, k):
    return self.comp[k].ECovMat()

  ######################################################### Local Params
  #########################################################  E-step
  ''' Inherited directly from ObsModel.py
  '''

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, SS, LP, **kwargs):
    ''' Calculate suff stats for the covariance matrix of each component
        xxT[k] = E[ x * x.T ] where x is the col vector of each observation
               = sum_{n=1}^N r_nk * outer(x_n, x_n)
    '''
    sqrtResp = np.sqrt(LP['resp'])
    K = sqrtResp.shape[1]
    xxT = np.zeros((K, self.D, self.D))
    for k in xrange(K):
      xxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis]*Data.X )
    SS.setField('xxT', xxT, dims=('K','D','D'))
    return SS

  ######################################################### Global Params
  #########################################################  M-step
  def update_obs_params_EM( self, SS, **kwargs):
    for k in xrange(self.K):
      covMat  = SS.xxT[k]/SS.N[k]
      covMat  += self.min_covar * np.eye(self.D)
      self.comp[k] = ZMGaussDistr( Sigma=covMat )

  def update_obs_params_VB( self, SS, mergeCompA=None, **kwargs):
    if mergeCompA is None:
      for k in xrange(self.K):
        self.comp[k] = self.obsPrior.get_post_distr(SS, k)
    else:
      self.comp[mergeCompA] = self.obsPrior.get_post_distr(SS, mergeCompA)

  def update_obs_params_soVB( self, SS, rho, **kwargs):
    for k in xrange(self.K):
      Dstar = self.obsPrior.get_post_distr(SS, k)
      self.comp[k].post_update_soVB( rho, Dstar)

  def set_global_params(self, hmodel=None, Sigma=None, **kwargs):
    ''' Set global parameters to provided values
    '''
    if hmodel is not None:
      self.comp = [copy.deepcopy(c) for c in hmodel.obsModel.comp]
      self.K = hmodel.obsModel.K
    if Sigma is not None:
      self.K = Sigma.shape[0]
      self.comp = [None for k in range(self.K)]
      for k in range(self.K):
        self.comp[k] = ZMGaussDistr(Sigma=Sigma[k])

  ######################################################### Evidence
  ######################################################### 
  def calc_evidence( self, Data, SS, LP=None):
    if self.inferType == 'EM': 
      # handled in alloc model and aggregated in HModel
      return 0
    else:
      return self.E_logpX(SS) + self.E_logpPhi() - self.E_logqPhi()    

  def E_logpX( self, SS ):
    lpX = np.zeros( self.K )
    for k in range(self.K):
      if SS.N[k] == 0:
        continue
      lpX[k] = SS.N[k]*self.comp[k].ElogdetLam() - \
                 self.comp[k].E_traceLam( SS.xxT[k] )
    return 0.5*np.sum( lpX ) - 0.5*np.sum(SS.N)*self.D*LOGTWOPI
     
  def E_logpPhi( self ):
    return self.E_logpLam()
      
  def E_logqPhi( self ):
    return self.E_logqLam()  
    
  def E_logpLam( self ):
    lp = np.zeros( self.K) 
    for k in xrange( self.K ):
      lp[k] = 0.5*(self.obsPrior.v - self.D -1)*self.comp[k].ElogdetLam()
      lp[k] -= 0.5*self.comp[k].E_traceLam( cholS=self.obsPrior.cholinvW())
    return lp.sum() - self.K * self.obsPrior.get_log_norm_const()
 
  def E_logqLam( self ):
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = -1*self.comp[k].get_entropy()
    return lp.sum()


  ######################################################### I/O Utils
  #########################################################   for humans
  def get_name(self):
    return 'ZMGauss'
      
  def get_info_string(self):
    return 'Zero-mean Gaussian distribution'
      
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      return 'Wishart on precision matrix \Lam \n' + self.obsPrior.to_string()

  ######################################################### I/O Utils
  #########################################################   for machines
  def get_prior_dict( self ):
    if self.obsPrior is None:
      PDict = dict(min_covar=self.min_covar, name="NoneType")
    else:
      PDict = self.obsPrior.to_dict()
    return PDict
