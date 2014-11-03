'''
XData.py

Data object for holding a dense matrix X of real 64-bit floating point numbers,
Each row of X represents a single observation.

Example
--------
>> import numpy as np
>> from bnpy.data import XData
>> X = np.random.randn(1000, 3) # Create 1000x3 matrix
>> myData = XData(X)
>> print myData.nObs
1000
>> print myData.D
3
>> print myData.X.shape
(1000,3)
'''

import numpy as np
from .DataObj import DataObj
from .MinibatchIterator import MinibatchIterator

class XData(DataObj):
  
  @classmethod
  def read_from_mat(cls, matfilepath, nObsTotal=None, **kwargs):
    ''' Static Constructor for building an instance of XData from disk
    '''
    import scipy.io
    InDict = scipy.io.loadmat( matfilepath, **kwargs)
    if 'X' not in InDict:
      raise KeyError('Stored matfile needs to have data in field named X')
    return cls( InDict['X'], nObsTotal )
  
  def __init__(self, X, nObsTotal=None, TrueZ=None):
    ''' Create an instance of XData given an array
        Reallocation of memory may occur, to ensure that 
          X is a 2D numpy array with proper byteorder, contiguity, and ownership.
    '''
    X = np.asarray(X)
    if X.ndim < 2:
      X = X[np.newaxis,:]
    self.X = np.float64(X.newbyteorder('=').copy())
    
    self.set_dependent_params(nObsTotal=nObsTotal)
    self.check_dims()
    if TrueZ is not None:
      self.addTrueLabels(TrueZ)
    
  def addTrueLabels(self, TrueZ):
    ''' Adds a "true" discrete segmentation of this data,
        so that each of the nObs items have a single label
    '''
    assert self.nObs == TrueZ.size
    self.TrueLabels = TrueZ
  
  def to_minibatch_iterator(self, **kwargs):
    return MinibatchIterator(self, **kwargs)

  #########################################################  internal methods
  #########################################################   
  def set_dependent_params( self, nObsTotal=None): 
    self.nObs = self.X.shape[0]
    self.dim = self.X.shape[1]
    if nObsTotal is None:
      self.nObsTotal = self.nObs
    else:
      self.nObsTotal = nObsTotal
    
  def check_dims( self ):
    assert self.X.ndim == 2
    assert self.X.flags.c_contiguous
    assert self.X.flags.owndata
    assert self.X.flags.aligned
    assert self.X.flags.writeable
    
  #########################################################  DataObj operations
  ######################################################### 
  def select_subset_by_mask(self, mask, doTrackFullSize=True):
    ''' Creates new XData object by selecting certain rows (observations)
        If doTrackFullSize is True, 
          ensure nObsTotal attribute is the same as the full dataset.
    '''
    if doTrackFullSize:
        return XData(self.X[mask], nObsTotal=self.nObsTotal)
    return XData(self.X[mask])

  def add_data(self, XDataObj):
    ''' Updates (in-place) this object by adding new data
    '''
    if not self.dim == XDataObj.dim:
      raise ValueError("Dimensions must match!")
    self.nObs += XDataObj.nObs
    self.nObsTotal += XDataObj.nObsTotal
    self.X = np.vstack([self.X, XDataObj.X])

  def get_random_sample(self, nObs, randstate=np.random):
    nObs = np.minimum(nObs, self.nObs)
    mask = randstate.permutation(self.nObs)[:nObs]
    Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
    return Data

  #########################################################  I/O methods
  ######################################################### 
  def __str__(self):
    np.set_printoptions(precision=5)
    return self.X.__str__()
    
  def summarize_num_observations(self):
    return '  num obs: %d' % (self.nObsTotal)
