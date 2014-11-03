'''
BarsBursty.py

Loads Mike Bryant's old bursty topics dataset. 
TODO: Need to write code to generate a bursty topic dataset

'''
from bnpy.data import WordsData, AdmixMinibatchIterator
import os

data_dir = '/data/liv/liv-x/topic_models/data/bars/'
matfilepath = os.environ['BNPYDATADIR'] + 'bars_bnpy_burstyK6_train.mat'

if not os.path.exists(matfilepath):
    matfilepath = data_dir + 'bars_bnpy_burstyK6_train.mat'

def get_data(seed=8675309, nObsTotal=25000, **kwargs):
    ''' Grab data from matfile specified by matfilepath
    '''
    Data = WordsData.read_from_mat( matfilepath )
    Data.summary = get_data_info(Data.nDocTotal, Data.vocab_size)
    return Data

def get_minibatch_iterator(seed=8675309, nBatch=10, nObsBatch=None, nObsTotal=25000, nLap=1, allocModelName=None, dataorderseed=0, **kwargs):
    Data = WordsData.read_from_mat( matfilepath )
    DataIterator = AdmixMinibatchIterator(Data, nBatch=nBatch, nObsBatch=nObsBatch, nLap=nLap, dataorderseed=dataorderseed)
    DataIterator.summary = get_data_info(Data.nDocTotal, Data.vocab_size)
    return DataIterator

def get_data_info(D, V):
    return 'Bars Bursty Data. D=%d. VocabSize=%d' % (D,V)
