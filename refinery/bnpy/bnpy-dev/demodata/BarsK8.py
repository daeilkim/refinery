'''
BarsK8.py

Toy Bars data, with K=8 topics
4 horizontal, and 4 vertical.
'''
import numpy as np
from bnpy.data import WordsData, AdmixMinibatchIterator

Defaults = dict()
Defaults['nDocTotal'] = 2000
Defaults['nWordsPerDoc'] = 100

SEED = 8675309

# FIXED DATA GENERATION PARAMS
K = 8 # Number of topics
V = 16 # Vocabulary Size
gamma = 0.5 # hyperparameter over doc-topic distribution

# TOPIC by WORD distribution
topics = np.zeros( (K,V) )
topics[0,:] = [ 9, 9, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
topics[1,:] = [ 0, 0, 0, 0, 9, 9, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0]
topics[2,:] = [ 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 9, 9, 0, 0, 0, 0]
topics[3,:] = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 9, 9]
topics[4,:] = [ 8, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0]
topics[5,:] = [ 0, 8, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0]
topics[6,:] = [ 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 8, 0]
topics[7,:] = [ 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 8]

# Add "smoothing" term to each entry of the topic-word matrix
# With V = 16 and 8 sets of bars,
#  smoothMass=0.02 yields 0.944 probability of drawing "on topic" word
smoothMass = 0.02 * 8
topics += smoothMass
# Ensure each row of topics is a probability vector
for k in xrange(K):
    topics[k,:] /= np.sum(topics[k,:])
Defaults['topics'] = topics

# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.hstack([1.1*np.ones(K/2), np.ones(K/2)])
trueBeta /= trueBeta.sum()
Defaults['topic_prior'] = gamma * trueBeta

def get_data_info(**kwargs):
    if 'nDocTotal' in kwargs:
      nDocTotal = kwargs['nDocTotal']
    else:
      nDocTotal = Defaults['nDocTotal']
    return 'Toy Bars Data. Ktrue=%d. nDocTotal=%d.' % (K, nDocTotal)

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