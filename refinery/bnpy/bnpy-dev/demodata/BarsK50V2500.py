'''
BarsK50V2500.py

Toy Bars data, with K=50 topics and V=2500 vocabulary size.
25 horizontal bars, and 25 vertical vertical ones.

Generated via the standard LDA generative model
  see WordsData.CreateToyDataFromLDAModel for details.

Usage
---------
To visualize example documents, execute this file as a script 
>> python BarsK50V2500.py

To visualize document "1" from within Python
>> Data = BarsK50V2500.get_data(nDocTotal=5)
>> wid1 = Data.word_id[ Data.doc_range[0,0]:Data.doc_range[0,1] ]
>> wct1 = Data.word_count[ Data.doc_range[0,0]:Data.doc_range[0,1] ]
Make histogram with counts for each of the vocab word types
>> whist = np.zeros(Data.vocab_size)
>> whist[wid1] = wct1
# Plot it as a 2D image
>> whist2D = np.reshape( whist, (50, 50) )
>> pylab.imshow(whist2D, interpolation='nearest')

'''
import numpy as np
from bnpy.data import WordsData, AdmixMinibatchIterator
import Bars2D

SEED = 8675309
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 50 # Number of topics
V = 2500 # Vocabulary Size
gamma = 0.75 # hyperparameter over doc-topic distribution

Defaults = dict()
Defaults['nDocTotal'] = 2000
Defaults['nWordsPerDoc'] = 5 * V / (K/2)

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
    return 'Toy Bars Data. Ktrue=%d. nDocTotal=%d. Typically 2-4 bars per doc' % (K, nDocTotal)

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
