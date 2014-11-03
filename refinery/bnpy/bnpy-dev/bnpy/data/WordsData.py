'''
WordsData.py

Data object that represents word counts across a collection of documents.

Terminology
-------
* Vocab : The finite collection of possible words.  
    {apple, berry, cardamom, fruit, pear, walnut}
  We assume this set has a fixed ordering, so each word is associated 
  with a particular integer in the set 0, 1, ... vocab_size-1
     0: apple        3: fruit
     1: berry        4: pear
     2: cardamom     5: walnut
* Document : a collection of words, observed together from the same source
  For example: 
      "apple, berry, berry, pear, pear, pear, walnut"

* nDoc : number of documents in the current, in-memory dataset
* nDocTotal : total number of docs, in entire dataset (for online applications)
'''

from .AdmixMinibatchIterator import AdmixMinibatchIterator
from .DataObj import DataObj
import numpy as np
import scipy.sparse
from ..util import RandUtil

class WordsData(DataObj):

  ######################################################### Constructor
  #########################################################
  def __init__(self, word_id=None, word_count=None, doc_range=None,
             vocab_size=0, vocab_dict=None, 
             nDocTotal=None, TrueParams=None, **kwargs):
    ''' Constructor for WordsData object

        Args
        -------
        word_id : nDistinctWords-length vector 
                  entry i gives VocabWordID for distinct word i in corpus
        word_count : nDistinctWords-length vector
                  entry i gives count for word_id[i] in that document
        doc_range : nDoc x 2 matrix
                  doc_range[d,:] gives (start,stop) for document d
                  where start/stop index rows in word_id,word_count
        vocab_size : integer size of set of possible vocabulary words
        vocab_dict : dict mapping integer vocab ids to strings
        nDocTotal : int total size of the corpus 
                    (in case this obj represents a minibatch)
        TrueParams : None [default], or dict of attributes
    '''
    self.word_id = np.asarray(np.squeeze(word_id), dtype=np.uint32)
    self.word_count = np.asarray(np.squeeze(word_count), dtype=np.float64)
    self.doc_range = np.asarray(doc_range, dtype=np.uint32)
    self.vocab_size = int(vocab_size)
  
    self._set_corpus_size_attributes(nDocTotal)
    self._verify_attributes()
  
    # Save "true" parameters that generated toy-data, if provided
    if TrueParams is not None:
      self.TrueParams = TrueParams

    # Add dictionary of vocab words, if provided
    if vocab_dict is not None:
      self.vocab_dict = vocab_dict

  def _set_corpus_size_attributes(self, nDocTotal=None):
    ''' Sets nDoc, nObs, and nDocTotal attributes of this WordsData object

        Args
        -------
        nDocTotal : int size of total corpus 
                    if None, nDocTotal is set equal to nDoc
    '''
    self.nDoc = self.doc_range.shape[0]
    self.nObs = len(self.word_id)
    if nDocTotal is None:
      self.nDocTotal = self.nDoc
    else:
      self.nDocTotal = int(nDocTotal)

  def _verify_attributes(self):
    ''' Basic runtime checks to make sure dimensions are set correctly
         for attributes word_id, word_count, doc_range, etc.
    '''
    assert self.vocab_size > 0
    assert self.word_id.ndim == 1
    assert self.word_id.min() >= 0
    assert self.word_id.max() < self.vocab_size
    assert self.word_count.ndim == 1
    assert self.word_count.min() > 0
    assert self.nDoc == self.doc_range.shape[0]
    assert self.nObs == len(self.word_id)
    assert self.doc_range.shape[1] == 2
    assert np.all( self.doc_range[:-1,1] == self.doc_range[1:,0])


  ######################################################### Sparse matrix
  #########################################################  representations
  def to_sparse_matrix(self, doBinary=False):
    ''' Make sparse matrix counting vocab usage across all words in dataset

        Returns
        --------
        C : sparse (CSC-format) matrix, size nObs x vocab_size
             C[n,v] = word_count[n] iff word_id[n] = v
                      0 otherwise
             That is, each word token n is represented by one entire row
                      with only one non-zero entry: at column word_id[n]

    '''
    if hasattr(self, "__sparseMat__") and not doBinary:
      return self.__sparseMat__
    if hasattr(self, '__sparseBinMat__') and doBinary:
      return self.__sparseBinMat__

    indptr = np.arange(self.nObs+1) # define buckets for one entry per row
    if doBinary:
      self.__sparseBinMat__ = scipy.sparse.csc_matrix(
                        (np.ones(self.nObs), np.int64(self.word_id), indptr),
                        shape=(self.vocab_size, self.nObs))
      return self.__sparseBinMat__

    else:
      self.__sparseMat__ = scipy.sparse.csc_matrix(
                        (self.word_count, np.int64(self.word_id), indptr),
                        shape=(self.vocab_size, self.nObs))
      return self.__sparseMat__
  
  def to_sparse_docword_matrix(self, weights=None, thr=None, **kwargs):
    ''' Make sparse matrix counting vocab usage for each document in dataset
        Used for efficient initialization of global parameters.

        Returns
        -------
        C : sparse (CSR-format) matrix, of shape nDoc-x-vocab_size, where
            C[d,v] = total count of vocab word v in document d
    '''
    if hasattr(self, "__sparseDocWordMat__") and weights is None:
      return self.__sparseDocWordMat__
    row_ind = list()
    col_ind = list()
    doc_range = self.doc_range
    word_count = self.word_count
    for d in xrange(self.nDoc):
      numDistinct = doc_range[d,1] - doc_range[d,0]
      doc_ind_temp = [d]*numDistinct
      row_ind.extend(doc_ind_temp)
      col_ind.extend(self.word_id[doc_range[d,0]:doc_range[d,1]])
    if weights is None:
      weights = self.word_count
    else:
      if thr is not None:
        mask = np.flatnonzero(weights > thr)
        weights = weights[mask] * self.word_count[mask]
        row_ind = np.asarray(row_ind)[mask]
        col_ind = np.asarray(col_ind)[mask]
      else:
        weights = weights * self.word_count
    sparseDocWordmat = scipy.sparse.csr_matrix(
                               (weights, (row_ind,col_ind)),
                               shape=(self.nDoc, self.vocab_size), 
                               dtype=np.float64)
    if weights is None:
      self.__sparseDocWordMat__ = sparseDocWordmat
    return sparseDocWordmat

  def get_nObs2nDoc_mat(self):
    ''' Returns nDoc x nObs sparse matrix
    '''
    data = np.ones(self.nObs)
    # row_ind will look like 0000, 111, 22, 33333, 444, 55
    col_ind = np.arange(self.nObs)

    indptr = np.hstack([Data.doc_range[0,0], Data.doc_range[:,1]])
    return scipy.sparse.csr_matrix( (data, (row_ind, col_ind)),
                                    shape=(self.nDoc, self.nObs),
                                    dtype=np.float64)

  ######################################################### DataObj interface
  #########################################################  methods
  def to_minibatch_iterator(self, **kwargs):
    ''' Return AdmixMinibatchIterator for this WordsData object,
          so we can traverse subsets of this document collection.
        Args
        -------
          see AdmixMinibatchIterator
    '''
    return AdmixMinibatchIterator(self, **kwargs)
   
  def add_data(self, WData):
    ''' Append provided WordsData to the end of this dataset
    '''
    assert self.vocab_size == WData.vocab_size
    self.word_id = np.hstack([self.word_id, WData.word_id])
    self.word_count = np.hstack([self.word_count, WData.word_count])
    startLoc = self.doc_range[-1,1]
    self.doc_range = np.vstack([self.doc_range, startLoc + WData.doc_range])
    self.nDoc += WData.nDoc
    self.nObs += WData.nObs
    self.nDocTotal += WData.nDocTotal
    self._verify_attributes()

  def get_random_sample(self, nDoc, randstate=np.random, candidates=None):
    ''' Create WordsData object for random subsample of this dataset

        Args
        -----
        nDoc : number of documents to choose
        randstate : numpy random number generator

        Returns
        -------
        WordsData : bnpy WordsData instance, with at most nDoc documents
    '''
    if candidates is None:
      docMask = randstate.permutation(self.nDoc)[:nDoc]
    else:
      docMask = randstate.permutation(candidates)[:nDoc]
    return self.select_subset_by_mask(docMask=docMask,
                                                doTrackFullSize=False)

  def select_subset_by_mask(self, docMask=None, wordMask=None,
                                doTrackFullSize=True):
    ''' Returns WordsData object representing a subset of this object,
  
        Args
        -------
        docMask : None, or list of document ids to select
        wordMask : None, or list of words to select
                 each entry is an index into self.word_id

        doTrackFullSize : boolean indicator for whether output dataset
                           should retain nDocTotal size of this object,
                        or should be self-contained (nDoc=nDocTotal) 

        Returns
        --------
        WordsData object, where
            nDoc = number of documents in the subset (=len(mask))
            nObs = nDistinctWords in the subset of docs
            nDocTotal defines size of entire dataset (not subset)
    '''
    if docMask is None and wordMask is None:
      raise ValueError("Must provide either docMask or wordMask")

    if docMask is not None:
      nDoc = len(docMask)
      nObs = np.sum(self.doc_range[docMask,1] - self.doc_range[docMask,0])
      word_id = np.zeros(nObs)
      word_count = np.zeros(nObs)
      doc_range = np.zeros((nDoc,2))
  
      # Fill in new word_id, word_count, and doc_range
      startLoc = 0
      for d in xrange(nDoc):
        start,stop = self.doc_range[docMask[d],:]
        endLoc = startLoc + (stop - start)
        word_count[startLoc:endLoc] = self.word_count[start:stop]
        word_id[startLoc:endLoc] = self.word_id[start:stop]
        doc_range[d,:] = [startLoc,endLoc]
        startLoc += (stop - start)

    elif wordMask is not None:
      wordMask = np.sort(wordMask)
      nObs = len(wordMask)
      docIDs = self.getDocIDs(wordMask)
      uDocIDs = np.unique(docIDs)
      nDoc = uDocIDs.size
      doc_range = np.zeros((nDoc,2))

      # Fill in new word_id, word_count, and doc_range
      word_id =  self.word_id[wordMask]
      word_count = self.word_count[wordMask]
      startLoc = 0
      for dd in range(nDoc):
        nWordsInCurDoc = np.sum(uDocIDs[dd] == docIDs)
        doc_range[dd,:] = startLoc, startLoc + nWordsInCurDoc
        startLoc += nWordsInCurDoc           

    nDocTotal=None
    if doTrackFullSize:
      nDocTotal = self.nDocTotal
    return WordsData(word_id, word_count, doc_range, self.vocab_size,
                     nDocTotal=nDocTotal)

  def getDocIDs(self, wordLocs=None):
    ''' Retrieve document ids for all word tokens, 
        or for a particular subset (if specified)

        Args
        -------
        wordLocs : None or ndarray of integer locations in range (0, self.nObs)
  
        Returns
        -------
        docIDs : 1-dim ndarray of integer document ids in range (0, nDoc)
    '''
    # Retrieve for entire dataset
    if wordLocs is None:
      if hasattr(self, "__docid__"):
        return self.__docid__
      self.__docid__ = np.zeros(self.word_id.size, dtype=np.uint32)
      for dd in range(self.nDoc):
        self.__docid__[self.doc_range[dd,0]:self.doc_range[dd,1]] = dd
      return self.__docid__

    # Retrieve for specified subset
    docIDs = np.zeros(len(wordLocs))
    for dd in range(self.nDoc):
      if dd == 0:
        matchMask = wordLocs < self.doc_range[dd,1] 
      else:
        matchMask = np.logical_and(wordLocs < self.doc_range[dd,1],
                                 wordLocs >= self.doc_range[dd-1,1])
      docIDs[matchMask] = dd
    return docIDs     

  ######################################################### Text summary
  ######################################################### 
  def get_text_summary(self, doCommon=True):
    ''' Returns human-readable summary of this object
    '''
    if hasattr(self, 'summary') and doCommon:
      s = self.summary
    elif doCommon:
      s = " nDoc %d, vocab_size %d\n" % (self.nDoc, self.vocab_size)
    else:
      s = ''
    return s + self.get_doc_stats_summary()

  def get_doc_stats_summary(self, pRange=[0,5, 50, 95, 100]):
    ''' Returns human-readable string summarizing word-count statistics
          e.g. word counts for the smallest, largest, and median-length doc
    '''
    nDistinctWordsPerDoc = np.zeros(self.nDoc)
    nTotalWordsPerDoc = np.zeros(self.nDoc)
    for d in range(self.nDoc):
      drange = self.doc_range[d,:]
      nDistinctWordsPerDoc[d] = drange[1] - drange[0]
      nTotalWordsPerDoc[d] = self.word_count[drange[0]:drange[1]].sum()
    assert np.sum(nDistinctWordsPerDoc) == self.word_id.size
    assert np.sum(nTotalWordsPerDoc) == np.sum(self.word_count)
    s = ''
    for p in pRange:
      if p == 0:
        sp = 'min'
      elif p == 100:
        sp = 'max'
      else:
        sp = "%d%%" % (p)
      s += "%5s " % (sp)
    s += '\n'
    for p in pRange:
      s += "%5s " % ("%.0f" % (np.percentile(nDistinctWordsPerDoc, p)))    
    s += ' nDistinctWordsPerDoc\n'
    for p in pRange:
      s += "%5s " % ("%.0f" % (np.percentile(nTotalWordsPerDoc, p)))    
    s += ' nTotalWordsPerDoc'
    return s

  ######################################################### Create from MAT
  #########################################################  (class method)
  @classmethod
  def read_from_mat(cls, matfilepath, **kwargs):
    ''' Creates an instance of WordsData from Matlab matfile
    '''
    import scipy.io
    InDict = scipy.io.loadmat(matfilepath, **kwargs)
    return cls(**InDict)

  ######################################################### Create from DB
  #########################################################  (class method)
  @classmethod
  def read_from_db(cls, dbpath, sqlquery, vocab_size=None, nDocTotal=None):
    ''' Creates an instance of WordsData from an SQL database
    '''
    import sqlite3
    # Connect to sqlite database and retrieve results as doc_data
    conn = sqlite3.connect(dbpath)
    conn.text_factory = str
    result = conn.execute(sqlquery)
    doc_data = result.fetchall()
    conn.close()
  
    # Repackage the doc_data into word_id, word_count attributes
    word_id = list()
    word_count = list()
    nDoc = len(doc_data)
    doc_range = np.zeros((nDoc,2), dtype=np.uint32)
    ii = 0
    for d in xrange( nDoc ):
      # make sure we subtract 1 for word_ids since python indexes by 0
      temp_word_id = [(int(n)-1) for n in doc_data[d][1].split()]
      temp_word_count = [int(n) for n in doc_data[d][2].split()]
      word_id.extend(temp_word_id)
      word_count.extend(temp_word_count)
      nUniqueWords = len(temp_word_id)
      doc_range[d,:] = [ii, ii + nUniqueWords]
      ii += nUniqueWords
    return cls(word_id=word_id, word_count=word_count,
               doc_range=doc_range, vocab_size=vocab_size, nDocTotal=nDocTotal)

  ######################################################### Create Toy Data
  #########################################################  (class method)
  @classmethod
  def CreateToyDataSimple(cls, nDoc=10, nWordsPerDoc=10, 
                        vocab_size=12, **kwargs):
    ''' Creates a simple toy instance of WordsData (good for debugging)
        Args
        --------
        nDoc : int num of documents to create
        nWordsPerDoc : int num of distinct words in each document
        vocab_size : int size of vocabulary
    '''
    PRNG = np.random.RandomState(0)
    word_id = list()
    word_count = list()
    doc_range = np.zeros((nDoc, 2))
    for dd in range(nDoc):
        wID = PRNG.choice(vocab_size, size=nWordsPerDoc, replace=False)
        wCount = PRNG.choice(np.arange(1,5), size=nWordsPerDoc, replace=True)
        word_id.extend(wID)
        word_count.extend(wCount)
        start = nWordsPerDoc * dd
        doc_range[dd,:] = [start, start + nWordsPerDoc]
    return cls(word_id=word_id, word_count=word_count, 
              doc_range=doc_range, vocab_size=vocab_size)

  @classmethod
  def CreateToyDataFromLDAModel(cls, seed=101, 
                nDocTotal=None, nWordsPerDoc=None, 
                topic_prior=None, topics=None,
                **kwargs):
    ''' Generates WordsData dataset via LDA generative model,
          given specific global parameters

        Args
        --------
        topic_prior : K-length vector of positive reals,
                      \pi_d \sim \Dir( topic_prior )
        topics : KxV matrix of positive reals, where rows sum to one
                  topics[k,v] := probability of vocab word v in topic k
    '''
    PRNG = np.random.RandomState(seed)

    K = topics.shape[0]
    V = topics.shape[1]
    # Make sure topics sum to one
    topics = topics / topics.sum(axis=1)[:,np.newaxis]
    assert K == topic_prior.size
  
    doc_range = np.zeros((nDocTotal, 2))
    wordIDsPerDoc = list()
    wordCountsPerDoc = list()

    alphaLP = np.zeros((nDocTotal,K))
    respPerDoc = list()

    # startPos : tracks start index for current doc within corpus-wide lists
    startPos = 0
    for d in xrange(nDocTotal):
      # Draw topic appearance probabilities for this document
      alphaLP[d,:] = PRNG.dirichlet(topic_prior)

      # Draw the topic assignments for this doc
      ## Npercomp : K-vector, Npercomp[k] counts appearance of topic k
      Npercomp = RandUtil.multinomial(nWordsPerDoc, alphaLP[d,:], PRNG)

      # Draw the observed words for this doc
      ## wordCountBins: V x 1 vector, entry v counts appearance of word v
      wordCountBins = np.zeros(V)
      for k in xrange(K):
        wordCountBins += RandUtil.multinomial(Npercomp[k], 
                                      topics[k,:], PRNG)

      # Record word_id, word_count, doc_range
      wIDs = np.flatnonzero(wordCountBins > 0)
      wCounts = wordCountBins[wIDs]
      assert np.allclose( wCounts.sum(), nWordsPerDoc)
      wordIDsPerDoc.append(wIDs)
      wordCountsPerDoc.append(wCounts)
      doc_range[d,0] = startPos
      doc_range[d,1] = startPos + wIDs.size  
      startPos += wIDs.size
  
      # Record expected local parameters (LP)
      curResp = (topics[:, wIDs] * alphaLP[d,:][:,np.newaxis]).T      
      respPerDoc.append(curResp)

    word_id = np.hstack(wordIDsPerDoc)
    word_count = np.hstack(wordCountsPerDoc)

    respLP = np.vstack(respPerDoc)
    respLP /= respLP.sum(axis=1)[:,np.newaxis]

    TrueParams = dict(K=K, topics=topics, beta=topic_prior,
                      word_variational=respLP, alphaPi=alphaLP)
    return WordsData(word_id, word_count, doc_range, V,
                    nDocTotal=nDocTotal, TrueParams=TrueParams)


  ######################################################### Write to file
  #########################################################  (instance method)
  def WriteToFile_ldac(self, filepath, min_word_index=0):
    ''' Write contents of this dataset to plain-text file in "ldac" format.
        
        Args

        Returns
        -------
        None. Writes to file instead.

        Each line of file represents one document, and has format
        [U] [term1:count1] [term2:count2] ... [termU:countU]
    '''
    word_id = self.word_id
    if min_word_index > 0:
      word_id = word_id + min_word_index
    with open(filepath, 'w') as f:
      for d in xrange(self.nDoc):
        dstart = self.doc_range[d,0]
        dstop = self.doc_range[d,1]
        nUniqueInDoc = dstop - dstart
        idct_list = ["%d:%d" % (word_id[n], self.word_count[n]) \
                              for n in xrange(dstart, dstop)]
        docstr = "%d %s" % (nUniqueInDoc, ' '.join(idct_list)) 
        f.write(docstr + '\n')
