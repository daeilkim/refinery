''' 
ZMGaussDistr.py 

Zero-mean, full-covariance Gaussian

Attributes
--------
Choose either covariance or precision representation.

Covariance: DxD matrix Sigma
Precision: DxD matrix L

Matrices must *always* be symmetric, and positive definite.
'''
import numpy as np
import scipy.linalg

from bnpy.util import dotATA, dotABT, dotATA
from bnpy.util import MVgammaln, MVdigamma, gammaln, digamma
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS

from Distr import Distr

class ZMGaussDistr( Distr ):

  def __init__( self, L=None, Sigma=None):
    if Sigma is not None:
      self.doSigma = True
      self.Sigma = np.asarray( Sigma )
      self.D = self.Sigma.shape[0]
    elif L is not None:
      self.doSigma = False
      self.L = np.asarray( L )
      self.D = self.L.shape[0]
    else:
      raise ValueError("Need to specify L or Sigma")
    self.Cache = dict()
  

  ######################################################### Log Cond. Prob.
  #########################################################  E-step
  def log_pdf( self, Data ):
    ''' Returns log p( x | theta )
    '''
    return -1*self.get_log_norm_const() - 0.5*self.dist_mahalanobis(Data.X)

  def dist_mahalanobis(self, X):
    '''  Given NxD matrix X, compute  Nx1 vector Dist
            Dist[n] = ( X[n]-m )' L (X[n]-m)
    '''
    if self.doSigma:
      Q = np.linalg.solve(self.cholSigma(), X.T)
    else:
      Q = dotABT(self.cholL(), X)
    Q *= Q
    return np.sum(Q, axis=0)

  ######################################################### Global Updates 
  #########################################################  M-step
  ''' None.  This class is for EM only, M-step handled by ZMGaussObsModel.py
  '''
  
    
  ######################################################### Basic properties
  ######################################################### 
  def get_log_norm_const( self ):
    ''' Returns log( Z ), where
         PDF(x) :=  1/Z(theta) f( x | theta )
    '''
    if self.doSigma:
      return 0.5*self.D*LOGTWOPI + 0.5*self.logdetSigma()
    else:
      return 0.5*self.D*LOGTWOPI - 0.5*self.logdetL()
    
  def get_entropy( self ):
    ''' Returns entropy of this distribution 
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx
    '''
    return self.get_log_norm_const() + 0.5*self.D
    

  ######################################################### Accessors
  ######################################################### 
  def ECovMat(self):
    if self.doSigma: 
      return self.Sigma
    else:
      return np.linalg.inv( self.L )
      
  def cholSigma(self):
    try:
      return self.Cache['cholSigma']
    except KeyError:
      self.Cache['cholSigma'] = scipy.linalg.cholesky(self.Sigma, lower=True )
      return self.Cache['cholSigma']
      
  def logdetSigma(self):
    try:
      return self.Cache['logdetSigma']
    except KeyError:
      self.Cache['logdetSigma'] = 2.0*np.sum(np.log(np.diag(self.cholSigma() )))
    return self.Cache['logdetSigma']
     
  def cholL(self):
    try:
      return self.Cache['cholL']
    except KeyError:
      self.Cache['cholL'] = scipy.linalg.cholesky(self.L, lower=False)
    return self.Cache['cholL']
          
  def logdetL(self):
    try:
      return self.Cache['logdetL']
    except KeyError:
      self.Cache['logdetL'] = 2.0*np.sum(np.log(np.diag(self.cholL())))
    return self.Cache['logdetL']


  ######################################################### I/O Utils 
  #########################################################
  def to_dict(self):
    if self.doSigma:
      return dict( Sigma=self.Sigma, name=self.__class__.__name__ )
    else:
      return dict( L=self.L, name=self.__class__.__name__ )
      
  def from_dict(self, Dict):
    if 'L' in Dict:
      self.L = Dict['L']
      self.doSigma = False
    elif 'Sigma' in Dict:
      self.Sigma = Dict['Sigma']
      self.doSigma = True
