'''
BetaDistr.py

Beta distribution in 1-dimension x ~ Beta(a,b)
    
Attributes
-------
lamA : pseudo count for # of heads
lamB : pseudo count for # of tails
  
'''
import numpy as np
import scipy.linalg
from scipy.special import digamma, gammaln

class BetaDistr(object):
    @classmethod
    def InitFromData(cls, argDict, Data):
        # Data should contain information about Beta vocabulary size
        if argDict["lamA"] and argDict["lamB"] is not None:
            lamA = argDict["lamA"]
            lamB = argDict["lamB"]
        else:
            lamA = 0.1
            lamB = 0.1
        return cls(lamA = lamA, lamB = lamB)
      
    def __init__(self, lamA=None, lamB=None, **kwargs):
        self.lamA = lamA
        self.lamB = lamB
        if lamA and lamB is not None:
            self.set_helpers(**kwargs)

    def set_helpers(self, doNormConstOnly=False, **kwargs):
        self.lamsum = self.lamA + self.lamB
        # What's the purpose of this?
        if hasattr(self, '_logNormC'):
          del self._logNormC
        if not doNormConstOnly:
          digammalamA = digamma(self.lamA)
          digammalamB = digamma(self.lamB)
          digammalamsum = digamma(self.lamA + self.lamB)
          self.ElogLamA = digammalamA - digammalamsum
          self.ElogLamB = digammalamB - digammalamsum

    ############################################################## Param updates  
    def get_post_distr(self, SS, k=None, kB=None, **kwargs):
        ''' Create new Distr object with posterior params'''
        if kB is not None:
          return BetaDistr(SS.sb_ss1 + self.lamA, SS.sb_ss0 + self.lamB, **kwargs)
        else:
          return BetaDistr(SS.sb_ss1 + self.lamA, SS.sb_ss0 + self.lamB, **kwargs)

    def post_update_soVB( self, rho, starD):
        ''' Stochastic online update of internal params'''
        self.lamA = rho * starD.lamA + (1.0 - rho) * self.lamA
        self.lamB = rho * starD.lamB + (1.0 - rho) * self.lamB
        self.set_helpers()

  ######################################################### Norm Constants
  #########################################################
    @classmethod
    def calc_log_norm_const(cls, lamA, lamB):
        return gammaln(lamA + lamB) - gammaln(lamA) - gammaln(lamB)

    def get_log_norm_const(self):
        ''' Returns log( Z ), where PDF(x) :=  1/Z(theta) f( x | theta )'''
        if hasattr(self, '_logNormC'):
          return self._logNormC
        self._logNormC = gammaln(self.lamA + self.lamB) - gammaln(self.lamA) - gammaln(self.lamB)
        return self._logNormC

    def get_entropy( self ):
        ''' Returns entropy of this distribution
          H[ p(x) ] = -1*\int p(x|theta) log p(x|theta) dx
        '''
        H = self.get_log_norm_const()
        H -= np.inner(self.lamvec - 1., self.Elogphi)
        return H


    ####################################################### I/O  
    #######################################################
    def to_dict(self):
        return dict(name=self.__class__.__name__, lamA=self.lamA, lamB=self.lamB)
    
    def from_dict(self, PDict):
        self.lamA = PDict['lamA']
        self.lamB = PDict['lamB']
        self.set_helpers()
