'''
DiagGaussObsModel.py

Multivariate, full-mean, diagonal-covariance Gaussian observation model.

See Also
--------
GaussObsModel, for full-covariance Gaussians.
'''
import numpy as np
import scipy.linalg
import os

from ..distr import GaussDistr
from ..distr import GaussGammaDistr

from ..util import LOGTWO, LOGPI, LOGTWOPI, EPS
from ..util import dotATA, dotATB, dotABT
from ..util import MVgammaln, MVdigamma

from ObsModel import ObsModel

class DiagGaussObsModel( ObsModel ):

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
      obsPrior = GaussGammaDistr.CreateAsPrior(priorArgDict,Data)
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
        self.comp[k] = GaussGammaDistr(**compDictList[k]) 
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
      return np.diag(self.comp[kk].b / self.comp[kk].a)


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
              xx : K x D x D array of "sums of squares"
    '''
    X = Data.X
    resp = LP['resp']
    K = resp.shape[1]
    
    # Expected mean for each k
    SS.setField('x', dotATB(resp, X), dims=('K', 'D'))
    # Expected covar for each k 
    SS.setField('xx', dotATB(resp, np.square(X)), dims=('K', 'D'))
    return SS
    
  ######################################################### Global Params
  #########################################################  M-step
  def update_obs_params_EM( self, SS, **kwargs):
    for k in xrange(self.K):
      mean    = SS.x[k]/SS.N[k]
      covMat_diag  = SS.xx[k]/SS.N[k] - np.square(mean)
      covMat_diag  += self.min_covar
      precMat = np.diag(1.0 / covMat_diag)
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

  ######################################################### Evidence  
  ######################################################### 
  def calc_evidence( self, Data, SS, LP):
    if self.inferType == 'EM':
     return 0 # handled by alloc model
    else:
      return self.E_logpX(SS) + self.E_logpPhi() - self.E_logqPhi()
   
  def E_logpX(self, SS):
    ''' E_q [ log p(X | Z, Phi) ]
    '''
    D = self.D
    lpX = np.zeros(self.K)
    for k in range(self.K):
      if SS.N[k] < 1e-9:
        continue
      ElogLam_N = self.comp[k].E_sumlogLam() * SS.N[k]
      ELam_S2 = np.inner(self.comp[k].E_Lam(), SS.xx[k])
      ELamMu_S = np.inner(self.comp[k].E_LamMu(), SS.x[k])
      ELamMu2_N = np.sum(self.comp[k].E_LamMu2()) * SS.N[k]
      lpX[k] = ElogLam_N - (ELam_S2 - 2*ELamMu_S + ELamMu2_N)
    logNormC = -0.5 * D * np.sum(SS.N) * LOGTWOPI
    return logNormC + 0.5 * np.sum(lpX)

  def E_logpPhi(self):
    '''
    '''
    logPDFConst = -1. * self.obsPrior.get_log_norm_const()
    Elogp = logPDFConst * np.ones(self.K)
    for k in xrange(self.K):
      Elogp[k] += self.comp[k].E_log_pdf_Phi(self.obsPrior, doNormConst=False)
    return np.sum(Elogp)

  def E_logqPhi(self):
    '''
    '''
    Elogq = np.zeros(self.K)
    for k in xrange(self.K):
      Elogq[k] = self.comp[k].E_log_pdf_Phi(self.comp[k], doNormConst=True)
    return np.sum(Elogq)
  
  ######################################################### I/O Utils
  #########################################################   for humans
  def get_name(self):
    return 'Diagonal Gauss'

  def get_info_string(self):
    return 'Gaussian distribution with diagonal covariance matrix'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      return 'Gaussian-Gamma jointly on \mu,\Lam\n'+ self.obsPrior.to_string()

  ######################################################### I/O Utils
  #########################################################   for machines
  def get_prior_dict( self ):
    if self.obsPrior is None:
      PDict = dict(min_covar=self.min_covar, name="NoneType")
    else:
      PDict = self.obsPrior.to_dict()
    return PDict
 
