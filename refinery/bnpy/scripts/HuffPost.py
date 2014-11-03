'''
HuffPost.py

'''
from bnpy.data import WordsData, AdmixMinibatchIterator
import os

data_dir = '/Users/daeil/Dropbox/research/bnpy/data/huffpost/'
matfilepath = os.environ['BNPYDATADIR'] + 'huffpost_bnpy.mat'

if not os.path.exists(matfilepath):
    matfilepath = data_dir + 'huffpost_bnpy.mat'

def get_data(seed=8675309, **kwargs):
    ''' Grab data from matfile specified by matfilepath
    '''
    Data = WordsData.read_from_mat(matfilepath)
    Data.summary = get_data_info(Data.nDocTotal, Data.vocab_size)
    return Data

def get_minibatch_iterator(seed=8675309, nBatch=10, nLap=1, 
                           dataorderseed=0, **kwargs):
    Data = WordsData.read_from_mat(matfilepath)
    DataIterator = AdmixMinibatchIterator(Data, nBatch=nBatch, nObsBatch=300, nLap=nLap, dataorderseed=dataorderseed)
    DataIterator.summary = get_data_info(Data.nDocTotal, Data.vocab_size)
    return DataIterator

def get_data_info(D, V):
    return 'Huffington Post Data. D=%d. VocabSize=%d' % (D,V)