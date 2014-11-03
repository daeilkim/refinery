'''
BarsK6V9.py

Toy Bars data, with K=6 topics and vocabulary size 9.
3 horizontal bars, and 3 vertical bars.

Generated via the standard LDA generative model
  see WordsData.CreateToyDataFromLDAModel for details.
'''
import numpy as np
from bnpy.data import WordsData, AdmixMinibatchIterator
import Bars2D

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 6 # Number of topics
V = 9 # Vocabulary Size
gamma = 0.5 # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 200
Defaults['nWordsPerDoc'] = 25

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.ones(K)
trueBeta /= trueBeta.sum()
Defaults['topic_prior'] = gamma * trueBeta

# TOPIC by WORD distribution
Defaults['topics'] = Bars2D.Create2DBarsTopicWordParams(V, K, PRNG=PRNG)

def get_data_info(**kwargs):
    if 'nDocTotal' in kwargs:
      nDocTotal = kwargs['nDocTotal']
    else:
      nDocTotal = Defaults['nDocTotal']
    return 'Toy Bars Data. Ktrue=%d. nDocTotal=%d. Typically 1-3 bars per doc.' % (K, nDocTotal)

def get_data(**kwargs):
    ''' 
        Args
        -------
        seed
        nDocTotal
        nWordsPerDoc
    '''
    Data = CreateToyDataFromLDAModel(seed=SEED, **kwargs)
    Data.summary = get_data_info(**kwargs)
    return Data

def get_minibatch_iterator(seed=SEED, nBatch=10, nLap=1,
                           dataorderseed=0, **kwargs):
    '''
        Args
        -------
        seed
        nDocTotal
        nWordsPerDoc
    '''
    Data = CreateToyDataFromLDAModel(seed=seed, **kwargs)
    DataIterator = AdmixMinibatchIterator(Data, 
                        nBatch=nBatch, nLap=nLap, dataorderseed=dataorderseed)
    DataIterator.summary = get_data_info(**kwargs)
    return DataIterator

def CreateToyDataFromLDAModel(**kwargs):
  for key in Defaults:
    if key not in kwargs:
      kwargs[key] = Defaults[key]
  return WordsData.CreateToyDataFromLDAModel(**kwargs)

if __name__ == '__main__':
  import bnpy.viz.BarsViz
  WData = CreateToyDataFromLDAModel(seed=SEED)
  bnpy.viz.BarsViz.plotExampleBarsDocs(WData)