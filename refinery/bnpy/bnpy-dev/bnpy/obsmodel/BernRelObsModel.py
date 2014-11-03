''' 
BagOfWordsObsModel.py
'''
import numpy as np
import copy
from scipy.special import digamma, gammaln

from ..util import np2flatstr, EPS
from ..distr import BetaDistr
from ObsModel import ObsModel


class BernRelObsModel(ObsModel):
    
  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType, obsPrior=None):
    self.inferType = inferType
    self.obsPrior = obsPrior
    self.epsilon = 1e-3
    self.comp = list()

  @classmethod
  def CreateWithPrior(cls, inferType, priorArgDict, Data):
    if inferType == 'EM':
      raise NotImplementedError('TODO')
    else:
      obsPrior = BetaDistr.InitFromData(priorArgDict, Data)
    return cls(inferType, obsPrior)

  @classmethod
  def CreateWithAllComps(cls, oDict, obsPrior, compDictList):
    ''' Create MultObsModel, all K component Distr objects,
         and the prior Distr object in one call
    '''
    if oDict['inferType'] == 'EM':
      raise NotImplementedError("TODO")

    self = cls(oDict['inferType'], obsPrior=obsPrior)
    self.K = len(compDictList)
    self.comp = [None for k in range(self.K)]
    for k in xrange(self.K):
      self.comp[k] = BetaDistr(**compDictList[k])
    return self

  ######################################################### Accessors
  #########################################################  
  def Elogphi(self):
    digLam = np.empty(self.Lam.shape)
    digamma(self.Lam, out=digLam)
    digLam -= digamma(self.LamSum)[:,np.newaxis]
    return digLam

  def ElogphiOLD(self):
    return digamma(self.Lam) - digamma(self.LamSum)[:,np.newaxis]

  ######################################################### Local Params
  #########################################################   E-step
  def calc_local_params(self, Data, LP, **kwargs):
    ''' Calculate local parameters (E-step)

        Returns
        -------
        LP : bnpy local parameter dict, with updated fields
        E_logsoftev_WordsData : nDistinctWords x K matrix, where
                               entry n,k = log p(word n | topic k)
    '''
    if self.inferType == 'EM':
      raise NotImplementedError('TODO')
    else:
      LP['E_logsoftev_EdgeLik'], LP['E_logsoftev_EdgeEps'] = self.E_logsoftev_EdgeData(Data, LP)
    return LP

  def E_logsoftev_EdgeData(self, Data, LP):
    ''' Return log soft evidence probabilities for each word token.

        Returns
        -------
        E_logsoftev_Edges : nDistinctEdges x K matrix
                                entry n,k gives E log p( edge_ij | community k)
    '''

    # Obtain matrix where col k = E[ log phi[k] ], for easier indexing
    N = Data.nNodeTotal # number of nodes
    E = Data.nEdgeTotal # number of distinct edges (1s and 0s)

    ElogLamA = np.zeros(self.K)
    ElogLamB = np.zeros(self.K)
    E_logsoftev_EdgeLik = np.zeros( (Data.nEdgeTotal, self.K) ) # actual responsibilities for edges
    E_logsoftev_EdgeEps = np.zeros( Data.nEdgeTotal ) # actual responsibilities for edges

    ElogEps1 = np.log(self.epsilon)
    ElogEps0 = np.log(1-self.epsilon)

    for k in xrange(self.K):
      ElogLamA[k] = self.comp[k].ElogLamA
      ElogLamB[k] = self.comp[k].ElogLamB

    E_logsoftev_EdgeLik[Data.ind1,:] = ElogLamA
    E_logsoftev_EdgeLik[Data.ind0,:] = ElogLamB
    E_logsoftev_EdgeEps[Data.ind1] = ElogEps1
    E_logsoftev_EdgeEps[Data.ind0] = ElogEps0

    return (E_logsoftev_EdgeLik, E_logsoftev_EdgeEps)

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats(self, Data, SS, LP, **kwargs):
    ''' Calculate and return sufficient statistics.

        Returns
        -------
        SS : bnpy SuffStatDict object, with updated fields
                WordCounts : K x VocabSize matrix
                  WordCounts[k,v] = # times vocab word v seen with topic k
    '''
    ev = LP['edge_variational']
    E, K = ev.shape
    sb_ss0 = np.sum(ev[Data.ind0,:], axis=0)
    sb_ss1 = np.sum(ev[Data.ind1,:], axis=0)
    SS.setField('sb_ss0', sb_ss0, dims=('K'))
    SS.setField('sb_ss1', sb_ss1, dims=('K'))
    return SS

  ######################################################### Global Params
  #########################################################   M-step

  def update_obs_params_EM(self, SS, **kwargs):
    raise NotImplementedError("TODO")

  def update_obs_params_VB(self, SS, mergeCompA=None, **kwargs):
    if mergeCompA is None:
      self.Lam[mergeCompA] = SS.WordCounts[mergeCompA] + self.obsPrior.lamvec    
    else:
      self.Lam = SS.WordCounts + self.obsPrior.lamvec    

  def update_obs_params_soVB( self, SS, rho, **kwargs):
    raise NotImplementedError("TODO")  

  def set_global_params(self, hmodel=None, lamA=None, lamB=None, theta=None, **kwargs):
    ''' Set global params to provided values

        Params
        --------
        lamA and lamB: the global variational parameters for the stochastic block matrix
        theta: the global variational parameters for node-community memberships
    '''
    if hmodel is not None:
        self.K = hmodel.obsModel.K
        self.comp = copy.deepcopy(hmodel.obsModel.comp)
        return
    self.K = theta.shape[1]
    self.comp = list()

    # Initialize each community to have an equal amount of edges
    for k in range(self.K):
        lamAp = self.obsPrior.lamA + lamA[k]
        lamBp = self.obsPrior.lamB + lamB[k]
        self.comp.append(BetaDistr(lamAp, lamBp))


  ######################################################### Evidence
  #########################################################
  def calc_evidence(self, Data, SS, LP):
    elbo_pData = self.Elogp_Edges(SS, LP)
    elbo_pLam  = self.Elogp_Lam()
    elbo_qLam  = self.Elogq_Lam()
    return elbo_pData + elbo_pLam - elbo_qLam

  def Elogp_Edges(self, SS, LP):
    ''' This should be different depending on assorative / non-assortative model
    '''
    Elogp_Edges = np.sum(LP['edge_variational'] * LP['E_logsoftev_EdgeLik']) \
                + np.sum( (1 - np.sum(LP['edge_variational'], axis=1)) * LP['E_logsoftev_EdgeEps'])
    return Elogp_Edges

  def Elogp_Lam(self):
    ''' Get the log pdf of the beta for ELBO
    '''
    logNormC = gammaln(self.obsPrior.lamA + self.obsPrior.lamB) \
             - gammaln(self.obsPrior.lamA) - gammaln(self.obsPrior.lamB)
    logPDF = 0
    for k in xrange(self.K):
        logPDF += (self.obsPrior.lamA - 1.0)*self.comp[k].ElogLamA \
                + (self.obsPrior.lamB - 1.0)*self.comp[k].ElogLamB
    return np.sum(logPDF + (self.K * logNormC) )

  def Elogq_Lam(self):
    '''
    '''
    logNormC = 0
    logPDF = 0
    for k in xrange(self.K):
        logNormC += gammaln(self.comp[k].lamA + self.comp[k].lamB) \
                  - gammaln(self.comp[k].lamA) - gammaln(self.comp[k].lamB)

        logPDF += (self.comp[k].lamA - 1.0)*self.comp[k].ElogLamA \
                + (self.comp[k].lamB - 1.0)*self.comp[k].ElogLamB
    return -1.0 * (logNormC + logPDF)


  ######################################################### I/O Utils
  #########################################################  for humans
  def get_info_string(self):
    return 'Bernoulli distribution'
  
  def get_info_string_prior(self):
    if self.obsPrior is None:
      return 'None'
    else:
      lamA = self.obsPrior.lamA
      lamB = self.obsPrior.lamB
      return 'Beta, lambda_A %s, lambda_B %s' % (lamA,lamB)

  ######################################################### I/O Utils
  #########################################################  for machines
  def get_prior_dict( self ):
    return self.obsPrior.to_dict()
 
