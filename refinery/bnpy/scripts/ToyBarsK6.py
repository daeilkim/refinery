'''
ToyBarsK6.py

'''
import numpy as np

from bnpy.data import WordsData, AdmixMinibatchIterator
from collections import defaultdict

K = 6 # Number of topics
V = 9 # Vocabulary Size
D = 250 # number of total documents
Nperdoc = 100 #words per document
alpha = 0.5 # hyperparameter over document-topic distributions
lamda = 0.1 # hyperparameter over topic by word distributions

# Create topic by word distribution
true_tw = np.zeros( (K,V) )
true_tw[0,:] = [ 1, 1, 1, 0, 0, 0, 0, 0, 0]
true_tw[1,:] = [ 0, 0, 0, 1, 1, 1, 0, 0, 0]
true_tw[2,:] = [ 0, 0, 0, 0, 0, 0, 1, 1, 1]
true_tw[3,:] = [ 1, 0, 0, 1, 0, 0, 1, 0, 0]
true_tw[4,:] = [ 0, 1, 0, 0, 1, 0, 0, 1, 0]
true_tw[5,:] = [ 0, 0, 1, 0, 0, 1, 0, 0, 1]
# add prior
true_tw += lamda
# ensure that true_tw is a probability
for k in xrange(K):
    true_tw[k,:] /= np.sum( true_tw[k,:] )

# total number of observations
nObs = V * D

# 8675309, sounds like a phone number...
def get_data(seed=8675309, nObsTotal=25000, **kwargs):
    words_dict = get_BoW(seed)
    Data = WordsData( **words_dict )
    Data.summary = get_data_info()
    return Data

def get_minibatch_iterator(seed=8675309, nBatch=10, nObsBatch=None, nObsTotal=25000, nLap=1, allocModelName=None, dataorderseed=0, **kwargs):
    words_dict = get_BoW(seed)
    Data = WordsData( **words_dict )
    DataIterator = AdmixMinibatchIterator(Data, nBatch=nBatch, nObsBatch=nObsBatch, nLap=nLap, dataorderseed=dataorderseed)
    DataIterator.summary = get_data_info()
    return DataIterator

def get_BoW(seed):
    nObsTotal = 0
    doc_range = np.zeros( (D, 2) )
    nUniqueEntry = 0 # counter to calculate document id locations
    true_td = np.zeros( (K,D) ) # true document x topic proportions
    PRNG = np.random.RandomState( seed )
    WCD = list() # document based word count list (most efficient)
    unif = np.ones(K) / K
    for d in xrange( D ):
        # documents are either in 1 to 3 topics
        td_weights = PRNG.multinomial(3,unif) + .001
        true_td[:,d] = td_weights / td_weights.sum()
        #true_td[:,d] = PRNG.dirichlet( alpha*np.ones(K) ) 
        Npercomp = PRNG.multinomial( Nperdoc, true_td[:,d])
        temp_word_count = defaultdict( int )
        for k in xrange(K):
            wordCounts = PRNG.multinomial(  Npercomp[k], true_tw[k,:] )
            for (wordID,count) in enumerate(wordCounts):
                if count == 0: 
                    continue
                temp_word_count[wordID] += count
                nObsTotal += count
        nDistinctEntry = len( temp_word_count )
        WCD.append(temp_word_count)
        
        #start and stop ids for documents
        doc_range[d,0] = nUniqueEntry
        doc_range[d,1] = nUniqueEntry+nDistinctEntry  
        nUniqueEntry += nDistinctEntry
    
    word_id = np.zeros( nUniqueEntry )
    word_count = np.zeros( nUniqueEntry )
    ii = 0
    for d in xrange(D):
        for (key,value) in WCD[d].iteritems():
            word_id[ii] = int(key)
            word_count[ii] = value
            ii += 1
        
    #Insert all important stuff in myDict
    myDict = defaultdict()
    myDict["true_tw"] = true_tw
    myDict["true_td"] = true_td # true 
    myDict["true_K"] = K
    
    # Necessary items
    myDict["doc_range"] = doc_range
    myDict["word_id"] = word_id
    myDict["word_count"] = word_count
    myDict["vocab_size"] = V

    return myDict

def get_data_info():
    return 'Toy Bars Data. Ktrue=%d. D=%d.' % (K,D)