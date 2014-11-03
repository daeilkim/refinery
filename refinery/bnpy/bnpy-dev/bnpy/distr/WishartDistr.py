''' 
WishartDistr.py 

Wishart probability distribution object.

Attributes
-------
v : degrees of freedom
invW : scale matrix of size DxD
D  : dimension of the scale matrix and Data
'''
import numpy as np
import scipy.linalg
import scipy.io

from bnpy.util import dotABT, dotATA
from bnpy.util import MVgammaln, MVdigamma, gammaln, digamma
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS

from Distr import Distr

class WishartDistr( Distr ):

  ######################################################### Constructor  
  #########################################################
  def __init__( self, v=None, invW=None, **kwargs):
    ''' Creates Wishart distr with degrees-of-freedom v and scale matrix invW
    '''
    self.v = float(v)
    self.invW = np.asarray(invW)
    if self.invW.ndim > 2:
      self.invW = np.squeeze(self.invW)
    self.D = self.invW.shape[0]
    self.Cache = dict()
    assert self.invW.ndim == 2

  @classmethod
  def CreateAsPrior(cls, argDict, Data):
    ''' Creates Wishart prior for Gaussian params that generate Data.
        Returns WishartDistr object with same dimension as Data.
        Provided argDict specifies the expected covariance matrix.
    '''
    D = Data.dim
    v  = np.maximum( argDict['dF'], D+2)
    ESigma = cls.createECovMatFromUserInput(argDict, Data)    
    return cls( v=v, invW= ESigma*(v-D-1) )

  ######################################################### Log Cond. Prob.  
  #########################################################  E-step
  def E_log_pdf( self, Data ):
    ''' Returns E[ log p( x | theta ) ] under q(theta) <- this distr
    '''
    logp = 0.5*self.ElogdetLam() \
          -0.5*self.E_dist_mahalanobis( Data.X )
    return logp
    
  def E_dist_mahalanobis(self, X ):
    '''Calculate Mahalanobis distance to x
             dist(x) = dX'*E[Lambda]*dX
       If X is a matrix, we compute a vector of distances to each row vector
             dist(X)[n] = dX[n]'*E[Lambda]*dX[n]
    '''
    xWx = np.linalg.solve(self.cholinvW(), X.T)
    xWx *= xWx
    return self.v * np.sum(xWx, axis=0)


  ######################################################### Global Updates  
  #########################################################  M-step
  def get_post_distr(self, SS, k=None, kB=None, **kwargs):
    ''' Create new Distr object with posterior params
    '''
    if k is None:
      EN = SS.N
      ExxT = SS.xxT
    elif kB is not None:
      EN = SS.N[k] + SS.N[kB]
      ExxT = SS.xxT[k] + SS.xxT[kB]
    else:
      EN = SS.N[k]
      ExxT = SS.xxT[k]
    if EN == 0:
      return WishartDistr(v=self.v, invW=self.invW)
    v    = self.v + EN
    invW = self.invW + ExxT
    return WishartDistr(v=v, invW=invW)
    
  def post_update_soVB( self, rho, starD ):
    ''' Online update of internal params
    '''
    self.v = rho * starD.v + (1.0 - rho) * self.v
    self.invW = rho * starD.invW + (1.0 - rho) * self.invW
    self.Cache = dict()

  ######################################################### Basic properties  
  #########################################################
  @classmethod
  def calc_log_norm_const( self, logdetW, v, D):
    return 0.5*v*D*LOGTWO + MVgammaln(0.5*v, D) + 0.5*v*logdetW  
    
  def get_log_norm_const( self ):
    ''' Returns log(Z), where
         PDF(x) :=  1/Z(theta) f( x | theta )
    '''
    v = self.v # readability
    D = self.D
    return 0.5*v*D*LOGTWO + MVgammaln(0.5*v, D) + 0.5*v*self.logdetW() 

  def get_entropy( self ):
    ''' Returns entropy of this distribution 
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx
        Remember, entropy for continuous distributions can be negative
          e.g. see Bishop Ch. 1 Eq. 1.110 for Gaussian discussion
    '''
    v = self.v
    D = self.D
    H = self.get_log_norm_const() -0.5*(v-D-1)*self.ElogdetLam() + 0.5*v*D
    return H
   
  ######################################################### Accessors  
  #########################################################
  def cholinvW(self):
    try:
      return self.Cache['cholinvW']
    except KeyError:
      self.Cache['cholinvW'] = scipy.linalg.cholesky(self.invW,lower=True)
      return self.Cache['cholinvW']
      
  def logdetW(self):
    try:
      return self.Cache['logdetW']
    except KeyError:
      self.Cache['logdetW'] = -2.0*np.sum(np.log(np.diag(self.cholinvW())))
      return self.Cache['logdetW']
      
  def ElogdetLam(self):
    try:
      return self.Cache['ElogdetLam']
    except KeyError:
      ElogdetLam = MVdigamma(0.5*self.v,self.D) + self.D*LOGTWO +self.logdetW()
      self.Cache['ElogdetLam'] = ElogdetLam
      return ElogdetLam

  def ECovMat(self):
    try:
      return self.Cache['ECovMat']
    except KeyError:
      self.Cache['ECovMat'] = self.invW/(self.v-self.D-1)
      return self.Cache['ECovMat']
   
 
  def E_traceLam( self, S=None, cholS=None):
    '''Calculate trace( S* E[Lambda] ) in numerically stable way
    '''
    if cholS is not None:
      Q = scipy.linalg.solve_triangular(self.cholinvW(), cholS, lower=True)
      return self.v * np.sum(Q**2)
    return self.v*np.trace( np.linalg.solve(self.invW, S) )

  ######################################################### I/O Utils  
  #########################################################
  def to_string(self, offset='  '):
    ''' Returns string summary of this Wishart distribution
    '''
    if self.D > 2:
      sfx = ' ...'
    else:
      sfx = ''
    np.set_printoptions( precision=3, suppress=True)
    msg = offset + 'E[ CovMat[k] ] = \n'
    msg +=  str(self.ECovMat()[:2,:2]) + sfx
    msg = msg.replace('\n', '\n' + offset)
    return msg

  def to_dict(self):
    return dict(v=self.v, invW=self.invW, name=self.__class__.__name__)

  def from_dict(self, PDict):
    self.v = PDict['v']
    self.invW = PDict['invW']
    self.D = self.invW.shape[0]
    self.Cache = dict()


  ######################################################### Other Utils  
  #########################################################
  @classmethod
  def createECovMatFromUserInput(cls, argDict, Data):    
    ''' Create DxD matrix ESigma, to be expected covariance matrix
          under this distribution
    '''
    D = Data.dim
    if argDict['ECovMat'] == 'eye':
      ESigma = argDict['sF'] * np.eye(D)
    elif argDict['ECovMat'] == 'covdata':
      ESigma = argDict['sF'] * np.cov(Data.X.T, bias=1)
    elif argDict['ECovMat'] == 'fromtruelabels':
      ''' Set Cov. Matrix by empirical Bayes
            average the within-class sample covariances
      '''
      Zvals = np.unique(Data.TrueLabels)
      Kmax = len(Zvals)
      CovHat = np.zeros((Kmax,D,D))
      wHat = np.zeros(Kmax)
      for kID,kk in enumerate(Zvals):
        wHat[kID] = np.sum(Data.TrueLabels==kk)
        CovHat[kID] = np.cov(Data.X[Data.TrueLabels==kk].T, bias=1)
      wHat = wHat/np.sum(wHat)
      ESigma = 1e-8 * np.eye((D,D))
      for kID in range(Kmax):
        ESigma += wHat[kID]*CovHat[kID]
    else:
      raise ValueError( 'Unrecognized scale matrix name %s' %(smatname) )
    return ESigma


  ######################################################### DEPRECATED
  """
  def ELam(self):
    try:
      return self.Cache['ELam']
    except KeyError:
      self.Cache['ELam'] = self.v*np.linalg.solve(self.invW,np.eye(self.D))
      return self.Cache['ELam']
  """
  """
  def traceW( self, S):
    '''Calculate trace( S* self.W ) in numerically stable way
    '''
    return np.trace( np.linalg.solve(self.invW, S)  )
  """