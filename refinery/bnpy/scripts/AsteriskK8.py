'''
AsteriskK8.py

Simple toy dataset of 8 Gaussian components with full covariance.  
Generated data form an "asterisk" shape when plotted in 2D.
'''
import scipy.linalg
import numpy as np
from bnpy.util.RandUtil import rotateCovMat
from bnpy.data import XData, MinibatchIterator

###########################################################  Set Toy Parameters
K = 8
D = 2

w = np.asarray( [1., 2., 1., 2., 1., 2., 1., 2.] )
w = w/w.sum()

# Place means evenly spaced around a circle
Rad = 1.0
ts = np.linspace(0, 2*np.pi, K+1)
ts = ts[:-1]
Mu = np.zeros( (K,D))
Mu[:,0] = np.cos(ts)
Mu[:,1] = np.sin(ts)

# Create basic 2D cov matrix with major axis much longer than minor one
V = 1.0/16.0
SigmaBase = np.asarray([[ V, 0], [0, V/100.0]])

# Create several Sigmas by rotating this basic covariance matrix
Sigma = np.zeros( (K,D,D) )
for k in xrange(K):
  Sigma[k] = rotateCovMat(SigmaBase, k*np.pi/4.0)

# Precompute cholesky decompositions
cholSigma = np.zeros( Sigma.shape )
for k in xrange( K ):
  cholSigma[k] = scipy.linalg.cholesky( Sigma[k] )

###########################################################  Module util functions
def sample_data_from_comp( k, Nk, PRNG ):
  return Mu[k,:] + np.dot(cholSigma[k].T, PRNG.randn(D, Nk) ).T

def get_short_name( ):
  ''' Return short string used in filepaths to store solutions
  '''
  return 'AsteriskK8'

def get_data_info():
  return 'Asterisk Toy Data. Ktrue=%d. D=%d.' % (K,D)

###########################################################  Generate the data
def get_X(seed, nObsTotal):
  PRNG = np.random.RandomState( seed )
  trueList = list()
  Npercomp = PRNG.multinomial( nObsTotal, w )
  X = list()
  for k in range(K):
    X.append( sample_data_from_comp( k, Npercomp[k], PRNG) )
    trueList.append( k*np.ones( Npercomp[k] ) )
  X = np.vstack( X )
  TrueZ = np.hstack( trueList )
  permIDs = PRNG.permutation( X.shape[0] )
  X = X[permIDs]
  TrueZ = TrueZ[permIDs]
  return X, TrueZ

###########################################################  User-facing accessors
def get_data(seed=8675309, nObsTotal=25000, **kwargs):
  '''
    Args
    -------
    seed : integer seed for random number generator,
            used for actually *generating* the data
    nObsTotal : total number of observations for the dataset.

    Returns
    -------
      Data : bnpy XData object, with nObsTotal observations
  '''
  X, TrueZ = get_X(seed, nObsTotal)
  Data = XData(X=X, TrueZ=TrueZ)
  Data.summary = get_data_info()
  return Data
  
def get_minibatch_iterator(seed=8675309, dataorderseed=0, nBatch=10, nObsBatch=None, nObsTotal=25000, nLap=1, startLap=0, **kwargs):
  '''
    Args
    --------
    seed : integer seed for random number generator,
            used for actually *generating* the data
    dataorderseed : integer seed that determines
                     (a) how data is divided into minibatches
                     (b) order these minibatches are traversed

   Returns
    -------
      bnpy MinibatchIterator object, with nObsTotal observations
        divided into nBatch batches
  '''
  X, TrueZ = get_X(seed, nObsTotal)
  Data = XData(X=X)
  Data.summary = get_data_info()
  DataIterator = MinibatchIterator(Data, nBatch=nBatch, nObsBatch=nObsBatch, nLap=nLap, startLap=startLap, dataorderseed=dataorderseed)
  return DataIterator

