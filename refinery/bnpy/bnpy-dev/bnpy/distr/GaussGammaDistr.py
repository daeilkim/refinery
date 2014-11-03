''' 
GaussGammaDistr.py 

Joint Gaussian-Gamma distribution: D independent Gaussian-Gamma distributions

Attributes
--------
m : mean for Gaussian, length D
kappa : scalar precision parameter for Gaussian covariance

a : parameter for Gamma, vector length D
b : parameter for Gamma, vector length D
'''
import numpy as np
import scipy.linalg

from bnpy.util import MVgammaln, MVdigamma
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import gammaln, digamma

from .Distr import Distr

class GaussGammaDistr( Distr ):

  ######################################################### Constructor  
  #########################################################

  def __init__(self, a=None, b=None, m=None, kappa=None, **kwargs):
    ''' Create new GaussGammaDistr object, with specified parameter values

        Args
        -------
        a : numpy 1D array_like, length D
        b : numpy 1D array_like, length D
        m : numpy 1D array_like, length D
        kappa : float

        Returns 
        -------
        D : bnpy GaussGammaDistr object, with provided parameters
    '''
    # Unpack
    self.a = np.squeeze(np.asarray(a))
    self.b = np.squeeze(np.asarray(b))
    self.m = np.squeeze(np.asarray(m))
    self.kappa = float(kappa)

    # Dimension check
    assert self.b.ndim <= 1
    assert self.m.shape == self.b.shape
    assert self.a.shape == self.m.shape
    self.D = self.b.size
    self.Cache = dict()
    
  @classmethod
  def CreateAsPrior( cls, argDict, Data):
    ''' Creates Gaussian-Gamma prior for params that generate Data.
        Returns GaussGammaDistr object with same dimension as Data.
        Provided argDict specifies prior's expected mean and variance.
    '''
    D = Data.dim
    a0 = argDict['a0']
    b0 = argDict['b0']
    m0 = argDict['m0']
    kappa = argDict['kappa']
    m = m0 * np.ones(D)
    a = a0 * np.ones(D)
    b = b0 * np.ones(D)
    return cls(a=a, b=b, m=m, kappa=kappa)
    

  ######################################################### Log Cond. Prob.  
  #########################################################   E-step

  def E_log_pdf( self, Data ):
    ''' Calculate E[ log p( x_n | theta ) ] for each x_n in Data.X
        
        Args
        -------
        Data : bnpy XData object
                with attribute Data.X, numpy 2D array of size nObs x D

        Returns
        -------
        logp : numpy 1D array, length nObs
    '''
    logPDFConst = -0.5 * self.D * LOGTWOPI + 0.5 * np.sum(self.E_logLam())
    logPDFData = -0.5 * self.E_distMahalanobis(Data.X)
    return logPDFConst + logPDFData


  def E_distMahalanobis(self, X):
    ''' Calculate E[ (x_n - \mu)^T diag(\lambda) (x_n - mu) ]
          which has simple form due to diagonal structure.

        Args
        -------
        X : numpy array, nObs x D

        Returns
        -------
        dist : numpy 1D array, length nObs
                dist[n] = E[ (X[n] - \mu)^T diag(\lambda) (X[n] - mu) ]
                        = expected mahalanobis distance to observation n
    '''
    Elambda = self.a / self.b
    if X.ndim == 2:
      weighted_SOS = np.sum( Elambda * np.square(X - self.m), axis=1)
    else:
      weighted_SOS = np.sum(Elambda * np.square(X - self.m))
    weighted_SOS += self.D/self.kappa
    return weighted_SOS

  ######################################################### Param updates 
  ######################################################### (M step)
  def get_post_distr( self, SS, k=None, kB=None, **kwargs):
    ''' Create new GaussGammaDistr as posterior given sufficient stats
           for a particular component (or components)

        Args
        ------
        SS : bnpy SuffStatBag, with K components
        k  : int specifying component of SS to use.
              Range {0, 1, ... K-1}.
        kB : [optional] int specifying additional component of SS to use
              if provided, k-th and kB-th entry of SS are *merged* additively
              Range {0, 1, ... K-1}. 

        Returns
        -------
        D : bnpy.distr.GaussGammaDistr, with updated posterior parameters
    '''
    if k is None:
      EN = SS.N
      Ex = SS.x
      Exx = SS.xx
    elif kB is not None:
      EN = float(SS.N[k] + SS.N[kB])
      Ex = SS.x[k] + SS.x[kB]
      Exx = SS.xx[k] + SS.xx[kB]
    else:
      EN = float(SS.N[k])
      Ex = SS.x[k]
      Exx = SS.xx[k]
    kappa = self.kappa + EN
    m = (self.kappa * self.m + Ex) / kappa
    a = self.a + 0.5*EN
    b = self.b + 0.5*(Exx + self.kappa*np.square(self.m) - kappa*np.square(m))
    return GaussGammaDistr(a, b, m, kappa)
     
  def post_update_soVB( self, rho, refDistr, **kwargs):
    ''' In-place update of this GaussGammaDistr's internal parameters,
          via the stochastic online variational algorithm.
  
        Updates via interpolation between self and reference.
          self = self * (1-rho) + refDistr * rho

        Args
        -----
        rho : float, learning rate to use for the update
        refDistr : bnpy GaussGammaDistr, reference distribution for update

        Returns
        -------
        None. 
    '''
    etaCUR = self.get_natural_params()
    etaSTAR = refDistr.get_natural_params()
    etaNEW = list(etaCUR)
    for i in xrange(len(etaCUR)):
      etaNEW[i] = rho*etaSTAR[i] + (1-rho)*etaCUR[i]
    self.set_natural_params(tuple(etaNEW))

  ######################################################### Required accessors
  ######################################################### 
  @classmethod
  def calc_log_norm_const(cls, a, b, m, kappa):
    logNormConstNormal = 0.5 * D * (LOGTWOPI + np.log(kappa))
    logNormConstGamma  = np.sum(gammaln(a)) - np.inner(a, np.log(b))
    return logNormConstNormal + logNormConstGamma
  
  def get_log_norm_const(self):
    ''' Calculate log normalization constant (aka log partition function)
          for this Gauss-Gamma distribution.

        p(mu,Lam) = NormalGamma( mu, Lam | a, b, m, kappa)
                  = 1/Z f(mu|Lam) g(Lam), where Z is const w.r.t mu,Lam
        Normalization constant = Z = \int f() g() dmu dLam

        Returns
        --------
        logZ : float
    '''
    D = self.D
    a = self.a
    b = self.b
    logNormConstNormal = 0.5 * D * (LOGTWOPI - np.log(self.kappa))
    logNormConstGamma  = np.sum(gammaln(a)) - np.inner(a, np.log(b))
    return logNormConstNormal + logNormConstGamma
    
  def E_log_pdf_Phi(self, Distr, doNormConst=True):
    ''' Evaluate expectation of log PDF for given GaussGammaDistr

        Args
        -------
        Distr : bnpy GaussGammaDistr
        doNormConst : boolean, if True then Distr's log norm const is included

        Returns
        -------
        logPDF : float
    '''
    assert Distr.D == self.D
    selfELam = self.a / self.b
    logPDF = np.inner(Distr.a - 0.5, self.E_logLam()) \
                - np.inner(Distr.b, selfELam) \
                - 0.5 * Distr.kappa * self.E_distMahalanobis(Distr.m)
    if doNormConst:
      return logPDF - Distr.get_log_norm_const()
    return logPDF

  def get_entropy(self):
    ''' Calculate entropy of this Gauss-Gamma disribution,
    '''
    return -1.0 * self.E_log_pdf_Phi(self)
     
  def get_natural_params(self):
    '''
    '''
    t1 = self.a
    t2 = self.b + 0.5 * self.kappa * np.square(self.m)
    t3 = self.kappa * self.m    
    t4 = self.kappa
    etatuple = t1, t2, t3, t4
    return etatuple

  def set_natural_params(self, etatuple):
    self.a = etatuple[0]
    self.kappa = etatuple[3]
    self.m = etatuple[2]/self.kappa
    self.b = etatuple[1] - 0.5 * self.kappa * np.square(self.m)
    self.Cache = dict()

  ######################################################### Custom Accessors
  #########################################################
  def E_logLam(self):
    ''' E[ \log \lambda_d ]
        
        Returns
        -------
        1D array, length D
    '''
    return digamma(self.a) - np.log(self.b)

  def E_sumlogLam(self):
    ''' \sum_d E[ \log \lambda_d ]
    
        Returns
        -------
        float, scalar
    '''
    return np.sum(digamma(self.a) - np.log(self.b))

  def E_Lam(self):
    ''' E[ \lambda_d * \mu_d ]
        Returns vector, length D
    '''
    return (self.a / self.b)

  def E_LamMu(self):
    ''' E[ \lambda_d * \mu_d ]
        Returns vector, length D
    '''
    return (self.a / self.b) * self.m

  def E_LamMu2(self):
    ''' E[ \lambda_d * \mu_d * \mu_d ]
        Returns vector, length D
    '''
    return (self.a / self.b) * np.square(self.m) + 1./self.kappa

  ############################################################## I/O 
  ##############################################################
  def to_dict(self):
    ''' Convert attributes of this GaussGammaDistr into a dict
          useful for long-term storage to disk, pickling, etc.

        Returns
        -------
        Dict with entries for each named parameter: a, b, m, kappa
    '''
    return dict(name=self.__class__.__name__, \
                 m=self.m, kappa=self.kappa, a=self.a, b=self.b)
    
  def from_dict(self, Dict):
    ''' Internally set this GaussGammaDistr's parameters via provided dict

        Returns
        --------
        None.  This Distr's parameters set to new values.
    '''
    self.m = Dict['m']
    self.a = Dict['a']
    self.b = Dict['b']
    self.kappa = Dict['kappa']
    self.D = self.b.shape[0]
    self.Cache = dict()

  def to_string(self, offset="  "):
    Elam = self.a[:2] / self.b[:2]
    if self.D > 2:
      sfx = '...\n'
    else:
      sfx = '\n'
    np.set_printoptions(precision=3, suppress=False)
    msg  = offset + 'E[ mean \mu ]         ' + str(self.m[:2]) + sfx
    msg += offset + 'E[ precision \lambda ]' + str(Elam) + sfx
    return msg
