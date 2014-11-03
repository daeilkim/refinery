'''
HardDPMixModel.py
Bayesian parametric mixture model with a unbounded number of components K

'''
import numpy as np

from bnpy.allocmodel import DPMixModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import NumericHardUtil
from bnpy.util import gammaln, digamma, EPS

class HardDPMixModel(DPMixModel):

  def requireMergeTerms(self):
    return False

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
    LP['resp'] = NumericHardUtil.toHardAssignmentMatrix(lpr)
    assert np.allclose(LP['resp'].sum(axis=1), 1)
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
    return SS

  ######################################################### Evidence
  #########################################################
  def calc_evidence(self, Data, SS, LP=None ):
    '''
    '''
    evV = self.E_logpV() - self.E_logqV()

    evZq = 0
    if SS.hasAmpFactor():
      evZ = self.E_logpZ(SS) -  SS.ampF * evZq
    else:
      evZ = self.E_logpZ(SS) - evZq
    return evZ + evV
