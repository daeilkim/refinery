'''
GraphData.py

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

class GraphData(DataObj):

  ######################################################### Constructor
  #########################################################
  def __init__(self, edge_val=None, edge_exclude=None, nNodeTotal=None, TrueParams=None,  **kwargs):
    ''' Constructor for WordsData object

        Args
        -------
        edge_id : the source and receiver node ids that have present edges
        edge_weight : this might be a vector of all ones if binary graph, if it contains heldout, those are set to -1
        nNodeTotal : total number of nodes
        TrueParams : None [default], or dict of attributes
        edge_exclude: nDistinctEdges x 3 (row id, column id, value) of edges that shouldn't be used during inference
    '''
    edge_val = np.asarray(np.squeeze(edge_val), dtype=np.int16)
    self.nNodeTotal = int(nNodeTotal)
    self.nEdgeTotal = nNodeTotal * (nNodeTotal - 1) / 2
    self.nPosEdges = len(edge_val)
    self.nAbsEdges = self.nEdgeTotal - self.nPosEdges
    self.graphType = 'undirected'

    # If there are edges that need to be heldout, concatenate edge_id and edge_weights
    if edge_exclude is not None:
        self.nExcludeEdges = len(edge_exclude)
        ind0 = np.nonzero(edge_exclude[:,2]==0)
        ind1 = np.nonzero(edge_exclude[:,2]==1)
        edge_exclude[ind0,2] = -1 # excluded edge values originally 0 set to -1
        edge_exclude[ind1,2] = -2 # excluded edge values originally 1 set to -2
        edge_val_test = np.concatenate((edge_val, edge_exclude), axis=0)
        self.edge_val = edge_val_test
    else:
        self.edge_val = edge_val

    # Save "true" parameters that generated toy-data, if provided
    if TrueParams is not None:
      self.TrueParams = TrueParams

    # Create a sparse matrix for these edges to use in analysis (algorithm depends on this)
    self.sparseY = self.to_sparse_matrix()

  ######################################################### Create Toy Data
  def to_sparse_matrix(self):
    ''' Make sparse matrix counting vocab usage across all words in dataset

        Returns
        --------
        C : sparse (CSC-format) matrix, of shape nObs-x-vocab_size, where
             C[n,v] = word_count[n] iff word_id[n] = v
                      0 otherwise
             That is, each word token n is represented by one entire row
                      with only one non-zero entry: at column word_id[n]

    '''
    if hasattr(self, "__sparseMat__"):
      return self.__sparseMat__
    self.__sparseMat__ = scipy.sparse.csc_matrix(
                        (self.edge_val[:,2], ( np.int64(self.edge_val[:,0]), np.int64(self.edge_val[:,1]) ) ),
                        shape=(self.nNodeTotal, self.nNodeTotal))
    return self.__sparseMat__

  def get_edges_all(self):
    ''' Typically expensive, but for small graphs we can get an edge list for all pairwise edges
    This is an E (total number of edges) x 3 matrix
    Column 1 = row id
    Column 2 = col id
    Column 3 = edge_value (i.e 0,1 or (-1,-2) for heldout edges respectively
    '''
    E = self.nEdgeTotal
    N = self.nNodeTotal
    edges = np.zeros( (E,3) )
    e = 0
    for ii in xrange(N):
        for jj in xrange(ii+1,N):
            edges[e,:] = [ii,jj,self.sparseY[ii,jj]]
            e += 1
    np.asarray(np.squeeze(edges), dtype=np.int8)
    self.indTrain = np.asarray(np.squeeze(np.nonzero(edges[:,2] >= 0)))
    self.edges = edges[self.indTrain,:]
    self.ind0 = np.asarray(np.squeeze(np.nonzero(self.edges[:,2]==0)))
    self.ind1 = np.asarray(np.squeeze(np.nonzero(self.edges[:,2]==1)))
    self.nEdgeTotal = len(self.edges)
    self.nObs = self.nEdgeTotal # for VBLearnAlg.py
    print "Total number of training edges: " + str(self.nEdgeTotal) + "/" + str(E)
    print "Total number of zero/ones within training: " + "(" + str(len(self.ind0)) + "/" + str(len(self.ind1)) + ")"

    return (self.edges, self.ind0, self.ind1, self.indTrain)

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
    pass