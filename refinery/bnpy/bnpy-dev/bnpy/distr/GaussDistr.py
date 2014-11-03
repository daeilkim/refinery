'''
GaussianDistr.py 

Gaussian probability distribution

Attributes
-------
m : D-dim vector, mean
L : DxD matrix, precision matrix
'''
import numpy as np
import scipy.linalg
from ..util import dotABT, MVgammaln, MVdigamma, gammaln, digamma
from ..util import LOGTWOPI, EPS
from .Distr import Distr

class GaussDistr( Distr ):
      
  ######################################################### Constructor  
  #########################################################
  def __init__(self, m=None, L=None):
    self.m = np.asarray( m )  
    self.L = np.asarray( L )
    self.D = self.m.size
    self.Cache = dict()

  ######################################################### Log Cond. Prob.  
  #########################################################   E-step
  def log_pdf(self, Data):
    ''' Calculate log soft evidence for all data items under this distrib
        
        Returns
        -------
        logp : Data.nObs x 1 vector, where
                logp[n] = log p( Data[n] | self's mean and prec matrix )
    '''
    return -1*self.get_log_norm_const() - 0.5*self.dist_mahalanobis(Data.X)
  
  def dist_mahalanobis(self, X):
    '''  Given NxD matrix X, compute  Nx1 vector Dist
            Dist[n] = ( X[n]-m )' L (X[n]-m)
    '''
    Q = dotABT(self.cholL(), X-self.m)
    Q *= Q
    return np.sum(Q, axis=0)
    
  ######################################################### Global updates
  #########################################################   M-step
  ''' None required. M-step handled by GaussObsModel.py
  '''

  ######################################################### Basic properties
  ######################################################### 
  @classmethod
  def calc_log_norm_const( self, logdetL, D):
    return 0.5 * D * LOGTWOPI - 0.5 * logdetL

  def get_log_norm_const( self ):
    ''' Returns log( Z ), where
         PDF(x) :=  1/Z(theta) f( x | theta )
    '''
    return 0.5 * self.D * LOGTWOPI - 0.5 * self.logdetL()

  def get_entropy( self ):
    ''' Returns entropy of this distribution 
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx
        Remember, entropy for continuous distributions can be negative
          e.g. see Bishop Ch. 1 Eq. 1.110 for Gaussian discussion
    '''
    return self.get_log_norm_const() + 0.5*self.D
    
        
  ######################################################### Accessors  
  #########################################################
  def get_natural_params( self ):
    eta = self.L, np.dot(self.L,self.m)
    return eta 

  def set_natural_params( self, eta ):
    L, Lm = eta
    self.L = L
    self.m = np.linalg.solve(L, Lm) # invL*L*m = m
    self.Cache = dict()

  def get_covar(self):
    try:
      return self.Cache['invL']
    except KeyError:
      self.Cache['invL'] = np.linalg.inv( self.L )
      return self.Cache['invL']

  def cholL(self):
    try:
      return self.Cache['cholL']
    except KeyError:
      self.Cache['cholL'] = scipy.linalg.cholesky(self.L) #UPPER by default
      return self.Cache['cholL']

  def logdetL(self):
    try:
      return self.Cache['logdetL']
    except KeyError:
      logdetL = 2.0*np.sum( np.log( np.diag( self.cholL() ) )  )
      self.Cache['logdetL'] =logdetL
      return logdetL  
  
  ######################################################### I/O Utils 
  #########################################################
  def to_dict(self):
    return dict(m=self.m, L=self.L)
    
  def from_dict(self, GDict):
    self.m = GDict['m']
    self.L = GDict['L']
    self.D = self.L.shape[0]
    self.Cache = dict()