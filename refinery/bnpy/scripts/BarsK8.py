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
true_tw = np.zeros( (K,V) )
true_tw[0,:] = [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
true_tw[1,:] = [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
true_tw[2,:] = [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
true_tw[3,:] = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
true_tw[4,:] = [ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
true_tw[5,:] = [ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
true_tw[6,:] = [ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
true_tw[7,:] = [ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
# Add "smoothing" term to each entry of the topic-word matrix
# With V = 16 and 8 sets of bars,
#  smoothMass=0.02 yields 0.944 probability of drawing "on topic" word
smoothMass = 0.02
true_tw += smoothMass
# Ensure each row of true_tw is a probability vector
for k in xrange(K):
    true_tw[k,:] /= np.sum( true_tw[k,:] )
Defaults['TopicWordProbs'] = true_tw


# GLOBAL PROB DISTRIBUTION OVER TOPICS
trueBeta = np.hstack([1.1*np.ones(K/2), np.ones(K/2)])
trueBeta /= trueBeta.sum()
Defaults['docTopicParamVec'] = gamma * trueBeta

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
    Data = genWordsData(seed=SEED, **kwargs)
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
    Data = genWordsData(seed=seed, **kwargs)
    DataIterator = AdmixMinibatchIterator(Data, 
                        nBatch=nBatch, nLap=nLap, dataorderseed=dataorderseed)
    DataIterator.summary = get_data_info(**kwargs)
    return DataIterator

def genWordsData(**kwargs):
  for key in Defaults:
    if key not in kwargs:
      kwargs[key] = Defaults[key]
  return WordsData.genToyData(**kwargs)

if __name__ == '__main__':
  import bnpy.viz.BarsViz
  WData = genWordsData(seed=SEED)
  bnpy.viz.BarsViz.plotExampleBarsDocs(WData)