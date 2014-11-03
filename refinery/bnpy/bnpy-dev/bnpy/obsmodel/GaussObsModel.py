'''
GaussObsModel.py

Multivariate, full-mean, full covariance Gaussian observation model for bnpy.

Contains information for accessing and updating
* data-generating parameters for all K components 
* parameters for the prior distribution on these data-generating parameters
'''
import numpy as np
import scipy.linalg
import os

from ..distr import GaussDistr
from ..distr import GaussWishDistr

from ..util import LOGTWO, LOGPI, LOGTWOPI, EPS
from ..util import dotATA, dotATB, dotABT
from ..util import MVgammaln, MVdigamma

from ObsModel import ObsModel

class GaussObsModel( ObsModel ):

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
    ''' Create GaussObsModel and its prior distr.
        Returns object that does not yet have global parameters
           until init_global_params() is called.
        Until then, it has no components and can't be used in learn algs.
    '''
    D = Data.dim
    if inferType == 'EM':
      obsPrior = None
      return cls(inferType, D, obsPrior, min_covar=priorArgDict['min_covar'])
    else:
      obsPrior = GaussWishDistr.CreateAsPrior(priorArgDict,Data)
      return cls(inferType, D, obsPrior)

  @classmethod
  def CreateWithAllComps(cls, oDict, obsPrior, compDictList):
    ''' Create GaussObsCompSet, all K component Distr objects, 
        and the prior Distr object in one call
    '''
    if 'min_covar' in oDict:
      mc = oDict['min_covar']
      self = cls(oDict['inferType'], obsPrior=obsPrior, min_covar=mc)
    else:
      self = cls(oDict['inferType'], obsPrior=obsPrior)
    self.K = len(compDictList)
    self.comp = [None for k in range(self.K)]
    
    for k in xrange(self.K):
      if self.inferType == 'EM':
        self.comp[k] = GaussDistr(**compDictList[k])
      else:
        self.comp[k] = GaussWishDistr(**compDictList[k]) 
      self.D = self.comp[k].D
    return self
    
  
  ######################################################### Accessors  
  #########################################################  
  def get_mean_for_comp(self, kk):
    return self.comp[kk].m
    
  def get_covar_mat_for_comp(self, kk):
    if self.inferType =='EM':
      return np.linalg.inv(self.comp[kk].L)
    else:
      return self.comp[kk].invW / (self.comp[kk].dF - self.D - 1)
      

  ######################################################### Local Params
  #########################################################  E-step
  ''' All methods directly inherited from ObsModel
  '''

  ######################################################### Suff Stats 
  #########################################################
  def get_global_suff_stats( self, Data, SS, LP, **kwargs):
    ''' Calculate suff stats for the global parameter update
        Args
        -------
        Data : bnpy XData object
        SS : bnpy SuffStatDict object
        LP : dict of local params, with field
              resp : Data.nObs x K array whose rows sum to one
                      resp[n,k] = posterior prob of comp k for data item n
        
        Returns
        -------
        SS : SuffStat object, with new fields
              x : K x D array of component-specific sums
              xxT : K x D x D array of "sums of outer products"
                    analogous to a sum of squares, but for covar matrix
    '''
    X = Data.X
    resp = LP['resp']
    K = resp.shape[1]
    
    # Expected mean for each k
    SS.setField('x', dotATB(resp, X), dims=('K','D'))

    # Expected covar for each k 
    sqrtResp = np.sqrt(resp)
    xxT = np.zeros( (K, self.D, self.D) )
    for k in xrange(K):
      xxT[k] = dotATA(sqrtResp[:,k][:,np.newaxis]*Data.X )
    SS.setField('xxT', xxT, dims=('K','D','D'))
    return SS
    

  ######################################################### Global Params
  #########################################################  M-step
  def update_obs_params_EM( self, SS, **kwargs):
    I = np.eye(self.D)
    for k in xrange(self.K):
      mean    = SS.x[k] / SS.N[k]
      covMat  = SS.xxT[k] / SS.N[k] - np.outer(mean,mean)
      covMat  += self.min_covar * I      
      precMat = np.linalg.solve( covMat, I )
      self.comp[k] = GaussDistr(m=mean, L=precMat)
           				 
  def update_obs_params_VB( self, SS, mergeCompA=None, **kwargs):
    if mergeCompA is None:
      for k in xrange(self.K):
        self.comp[k] = self.obsPrior.get_post_distr(SS, k)
    else:
      self.comp[mergeCompA] = self.obsPrior.get_post_distr(SS, mergeCompA)

  def update_obs_params_soVB( self, SS, rho, **kwargs):
    for k in xrange(self.K):
      Dstar = self.obsPrior.get_post_distr(SS, k)
      self.comp[k].post_update_soVB(rho, Dstar)


  def set_global_params(self, hmodel=None, m=None, L=None, **kwargs):
    ''' Set global parameters to provided values
    '''
    if hmodel is not None:
      self.comp = [copy.deepcopy(c) for c in hmodel.obsModel.comp]
      self.K = hmodel.obsModel.K
    elif L is not None:
      self.K = L.shape[0]
      self.comp = [None for k in range(self.K)]
      for k in range(self.K):
        self.comp[k] = GaussDistr(m=m[k], L=L[k])

  ######################################################### Evidence  
  ######################################################### 
  def calc_evidence( self, Data, SS, LP):
    if self.inferType == 'EM':
     return 0 # handled by alloc model
    else:
      return self.E_logpX(LP, SS) + self.E_logpPhi() - self.E_logqPhi()
  
  def E_logpX( self, LP, SS ):
    ''' E_{q(Z), q(Phi)} [ log p(X) ]
        Bishop PRML eq. 10.71
    '''
    lpX = -self.D*LOGTWOPI*np.ones( self.K )
    for k in range(self.K):
      if SS.N[k] < 1e-9:
        lpX[k] += self.comp[k].ElogdetLam() - self.D/self.comp[k].kappa
      else:
        mean    = SS.x[k]/SS.N[k]
        covMat  = SS.xxT[k]/SS.N[k] - np.outer(mean,mean)
        lpX[k] += self.comp[k].ElogdetLam() - self.D/self.comp[k].kappa \
                - self.comp[k].dF* self.comp[k].traceW( covMat )  \
                - self.comp[k].dF* self.comp[k].dist_mahalanobis(mean )
    return 0.5*np.inner(SS.N,lpX)
    
  def E_logpPhi( self ):
    return self.E_logpLam() + self.E_logpMu()
      
  def E_logqPhi( self ):
    return self.E_logqLam() + self.E_logqMu()
  
  def E_logpMu( self ):
    ''' First four RHS terms (inside sum over K) in Bishop 10.74
    '''
    lp = np.empty( self.K)    
    for k in range( self.K ):
      mWm = self.comp[k].dist_mahalanobis( self.obsPrior.m )
      lp[k] = self.comp[k].ElogdetLam() \
                -self.D*self.obsPrior.kappa/self.comp[k].kappa \
                -self.obsPrior.kappa*self.comp[k].dF*mWm
    lp += self.D*( np.log( self.obsPrior.kappa ) - LOGTWOPI)
    return 0.5*lp.sum()
    
  def E_logpLam( self ):
    ''' Last three RHS terms in Bishop 10.74
    '''
    lp = np.empty( self.K) 
    for k in xrange( self.K ):
      lp[k] = 0.5*(self.obsPrior.dF - self.D -1)*self.comp[k].ElogdetLam()
      lp[k] -= 0.5*self.comp[k].dF*self.comp[k].traceW(self.obsPrior.invW)
    return lp.sum() - self.K * self.obsPrior.logWishNormConst()
    
  def E_logqMu( self ):
    ''' First two RHS terms in Bishop 10.77
    '''    
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] = 0.5*self.comp[k].ElogdetLam()
      lp[k] += 0.5*self.D*( np.log( self.comp[k].kappa ) - LOGTWOPI )
    return lp.sum() - 0.5*self.D*self.K
                     
  def E_logqLam( self ):
    ''' Last two RHS terms in Bishop 10.77
    '''
    lp = np.zeros( self.K)
    for k in xrange( self.K):
      lp[k] -= self.comp[k].entropyWish()
    return lp.sum()
 
  
  ######################################################### I/O Utils
  #########################################################   for humans
  def get_name(self):
    return 'Gauss'

  def get_info_string(self):
    return 'Gaussian distribution'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      msg = 'Gauss-Wishart jointly on mu[k], Lam[k] \n'
      return msg + self.obsPrior.to_string()


  ######################################################### I/O Utils
  #########################################################   for machines
  def get_prior_dict( self ):
    if self.obsPrior is None:
      PDict = dict(min_covar=self.min_covar, name="NoneType")
    else:
      PDict = self.obsPrior.to_dict()
    return PDict
 
