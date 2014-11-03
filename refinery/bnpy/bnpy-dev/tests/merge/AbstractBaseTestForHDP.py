
import numpy as np
import unittest

import bnpy
from bnpy.learnalg import MergeMove
from scipy.special import digamma
import copy

######################################################### Make Data
#########################################################
def MakeData(K=4, D=2500, nWordsPerDoc=50):
  ''' Simple 4 component data on 6 word vocabulary
      
  '''
  topics = np.zeros((K,6))
  topics[0] = [0.48, 0.48, 0.01, 0.01, 0.01, 0.01]
  topics[1] = [0.01, 0.01, 0.48, 0.48, 0.01, 0.01]
  topics[2] = [0.01, 0.01, 0.01, 0.01, 0.48, 0.48]
  topics[3] = [0.01, 0.33, 0.01, 0.32, 0.01, 0.32]
  topic_prior = 0.1 * np.ones(4)
  Data = bnpy.data.WordsData.CreateToyDataFromLDAModel(
                      topics=topics, topic_prior=topic_prior,
                      nDocTotal=D,
                      nWordsPerDoc=nWordsPerDoc, seed=123)
  trueResp = Data.TrueParams['word_variational']
  assert np.allclose(1.0,np.sum(trueResp,axis=1))
  return Data, trueResp  

def MakeMinibatches(Data):
  PRNG = np.random.RandomState(1234)
  permIDs =  PRNG.permutation(Data.nDocTotal)
  bIDs1 = permIDs[:len(permIDs)/2]
  bIDs2 = permIDs[len(permIDs)/2:]
  batchData1 = Data.select_subset_by_mask(bIDs1)
  batchData2 = Data.select_subset_by_mask(bIDs2)
  return batchData1, batchData2

######################################################### Make Data
#########################################################
class AbstractBaseTestForHDP(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    Data, trueResp = MakeData()
    batchData1, batchData2 = MakeMinibatches(Data)
    self.Data = Data
    self.trueResp = trueResp
    self.batchData1 = batchData1
    self.batchData2 = batchData2
    self.MakeModelWithTrueComps()

  ######################################################### Make Model
  #########################################################

  def MakeModelWithTrueComps(self):
    ''' Create model with true components that generated self.Data
    ''' 
    aDict = dict(alpha0=1.0, gamma=0.1)
    oDict = {'lambda':0.05}
    self.hmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult',
                                             aDict, oDict, self.Data)
    LP = self.getTrueLP()
    SS = self.hmodel.get_global_suff_stats(self.Data, LP)
    self.hmodel.update_global_params(SS)

  def MakeModelWithDuplicatedComps(self):
    ''' Create model with "duplicated" components,
          for each true comp that generated self.Data, 
          self.dupModel has two versions
    ''' 
    aDict = dict(alpha0=1.0, gamma=0.1)
    oDict = {'lambda':0.05}
    self.dupModel = bnpy.HModel.CreateEntireModel('VB', 'HDPModel', 'Mult',
                                             aDict, oDict, self.Data)
    dupLP = self.getDupLP()
    dupSS = self.dupModel.get_global_suff_stats(self.Data, dupLP)
    self.dupModel.update_global_params(dupSS)

  def getTrueLP(self):
    return self.getLPfromResp(self.trueResp)

  def getDupLP(self):
    ''' Create local parameters for "duplicated" model
          each comp k in true model is divided into two comps k1, k2
          any words with z[n] = k in "first half" of Data are assigned to k1
          any words with z[n] = k in "second half" are assigned to k2
    '''
    Data = self.Data
    K = self.trueResp.shape[1]
    dupResp = np.zeros((Data.nObs, 2*K))
    dupResp[:Data.nObs/2,:K] = self.trueResp[:Data.nObs/2]
    dupResp[Data.nObs/2:,K:] = self.trueResp[Data.nObs/2:]
    return self.getLPfromResp(dupResp)

  def getLPfromResp(self, Resp, smoothMass=0.001):
    ''' Create full local parameter (LP) dictionary for HDPModel,
          given responsibility matrix Resp

        Returns
        --------
        LP : dict with fields word_variational, alphaPi, E_logPi, DocTopicCount
    '''
    Data = self.Data
    D = Data.nDoc
    K = Resp.shape[1]
    # DocTopicCount matrix : D x K matrix
    DocTopicC = np.zeros((D, K))
    for dd in range(D):
      start,stop = Data.doc_range[dd,:]
      DocTopicC[dd,:] = np.dot(Data.word_count[start:stop],        
                               Resp[start:stop,:]
                               )
    assert np.allclose(DocTopicC.sum(), Data.word_count.sum())
    # Alpha and ElogPi : D x K+1 matrices
    padCol = smoothMass * np.ones((D,1))
    alph = np.hstack( [DocTopicC + smoothMass, padCol])    
    ElogPi = digamma(alph) - digamma(alph.sum(axis=1))[:,np.newaxis]
    assert ElogPi.shape == (D,K+1)
    return dict(word_variational =Resp, 
              E_logPi=ElogPi, alphaPi=alph,
              DocTopicCount=DocTopicC)    


  def run_Estep_then_Mstep(self):
    ''' Perform one full update to self.hmodel's global parameters
          given self.Data as observed data
        Runs Estep (LP), then SSstep (SS), then Mstep 

        Returns
        --------- 
        LP
        SS
    '''
    LP = self.hmodel.calc_local_params(self.Data)
    flagDict = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    SS = self.hmodel.get_global_suff_stats(self.Data, LP, **flagDict)
    self.hmodel.update_global_params(SS)
    return LP, SS