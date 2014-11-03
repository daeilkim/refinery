'''
NYTimes.py

'''
from bnpy.data import WordsData, AdmixMinibatchIteratorDB

dbpath = '/Users/daeil/Dropbox/research/local/nytimes_ldc'
''' Use the dbpath below in order to connect to the nytimes database at Brown
'''
#dbpath='/data/liv/nytimes/liv/nytimes_ldc'

D = 1816909
V = 8000

def get_data(seed=8675309, nObsTotal=25000, **kwargs):
    ''' Grab data from database to initialize (used only once really)
    '''
    doc_id_select = range(1,500) # grab the first 500 documents to initialize
    nDoc = len(doc_id_select)
    query = 'select * from data where rowid in (' + ','.join(map(str, doc_id_select)) + ')'
    Data = WordsData.read_from_db( dbpath, query, nDoc=nDoc, nDocTotal = nDoc, vocab_size = V )
    Data.summary = get_data_info(Data.nDocTotal, Data.vocab_size)
    return Data

def get_minibatch_iterator(seed=8675309, nBatch=10000, nObsBatch=None, nObsTotal=25000, nLap=1, allocModelName=None, dataorderseed=0, **kwargs):
    ''' Data is primarily loaded through AdmixMinibatchIteratorDB. 
    If creating from database, put in true number of documents and vocabulary size for the entire corpus
    Initialize with only a handful of documents however, specified by doc_id_select
    '''
    #Data object isn't passed in, is this bottom part necessary again?
    #doc_id_select = range(1,500) # grab the first 500 documents
    #query = 'select * from data where rowid in (' + ','.join(map(str, doc_id_select)) + ')'
    #Data = WordsData.read_from_db( dbpath, query, nDoc=len(doc_id_select), nDocTotal = D, vocab_size = V )
    Data = get_data(nDocTotal = D, vocab_size = V)
    
    #Create iterator that grabs documents from the sqlite3 database
    DataIterator = AdmixMinibatchIteratorDB(Data, dbpath=dbpath, nDocTotal=D, nBatch=nBatch, nObsBatch=nObsBatch, nLap=nLap, dataorderseed=dataorderseed)
    DataIterator.summary = get_data_info(D, V)
    return DataIterator

def get_data_info(D, V):
    return 'NYTimes (Very Large) Data. D=%d. VocabSize=%d' % (D,V)