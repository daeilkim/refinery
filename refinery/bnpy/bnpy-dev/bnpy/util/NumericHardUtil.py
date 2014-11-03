'''
NumericHardUtil.py

Library of efficient vectorized implementations of
 operations common to hard-assignment clustering/topic modeling

'''
import numpy as np
import scipy.sparse
from scipy.special import gammaln

def colwisesumLogFactorial_allpairs(Nmat):
  '''
  '''
  K = Nmat.shape[1]
  logFactMat = np.zeros((K,K))
  for jj in xrange(K-1):
    curNmat = Nmat[:,jj][:,np.newaxis] + Nmat[:, jj+1:]
    logFactMat[jj,jj+1:] = colwisesumLogFactorial(curNmat)
  return logFactMat

def colwisesumLogFactorial_specificpairs(Nmat, mPairs):
  '''
  '''
  K = Nmat.shape[1]
  logFactMat = np.zeros((K,K))
  for kA, kB in mPairs:
    logFactMat[kA, kB] = colwisesumLogFactorial(Nmat[:,kA] + Nmat[:,kB])
  return logFactMat

###########################################################
###########################################################
def toHardAssignmentMatrix(P, dtype=np.float64):
  ''' Convert "soft" log probability matrix to hard assignments
      
      Example
      ------
      >>> logpMat = np.asarray( [[-10, -20], [-33, -21]] )
      [ 1 0 
        0 1 ]
  '''
  N, K = P.shape
  colIDs = np.argmax(P, axis=1)
  Phard = scipy.sparse.csr_matrix(
              (np.ones(N, dtype=dtype), colIDs, np.arange(N+1)),
              shape=(N, K), dtype=dtype)
  return Phard.toarray()

def toHardAssignmentMatrix_direct(P, dtype=np.float64):
  ''' Convert "soft" log probability matrix to hard assignments
      
      Example
      ------
      >>> logpMat = np.asarray( [[-10, -20], [-33, -21]] )
      [ 1 0 
        0 1 ]
  '''
  N, K = P.shape
  Phard = np.zeros((N,K), dtype=dtype)
  rowIDs = np.arange(N, dtype=np.int64)
  colIDs = np.argmax(P, axis=1)
  ids = np.ravel_multi_index([rowIDs, colIDs], P.shape)
  Phard.ravel()[ids] = 1
  return Phard

###########################################################
###########################################################
def findMode_Mult(Nvec, Wmat):
  return findMode_Mult_fastest_skipsingles(Nvec, Wmat)

def findMode_Mult_1D(N, w):
    w = np.asarray(w, dtype=np.float64)
    n = np.floor( N * w)
    f = N * w - n
    q = (1 - f)/ w
    while n.sum() < N:
        minID = np.argmin(q)
        n[minID] += 1
        if n.sum() < N:
            q[minID] += 1/w[minID]            
    assert np.abs(n.sum() - N) < 0.0001
    return n

def findMode_Mult_loop(Nvec, Wmat):
  Nmat = np.zeros( Wmat.shape, dtype=np.int64)
  for row in xrange(Nvec.size):
    if Nvec[row] == 1:
      Nmat[row, np.argmax(Wmat[row])] = 1
    else:
      Nmat[row,:] = findMode_Mult_1D(Nvec[row], Wmat[row])
  return Nmat

def findMode_Mult_basic(Nvec, Wmat):
  ''' Find modes to multinomial with given parameters
        vectorized to solve many problems simultaneously

      Args
      ------
      Nvec : 1D array, size P
             contains non-negative integers
      Wmat : 2D array, size P x K
             each row is valid probability vector of length K
             Wmat[p] has non-negative entries, sums to one

      Returns
      -------
      Nmat : 2D sparse array, size P x K
          Nmat[p,:] = argmax_{n} \log p_{mult}( Nmat[p] | Nvec[p], Wmat[p] ) 
  '''
  # Unpack and parse input
  Nvec = np.asarray(Nvec, dtype=np.int64)
  if Nvec.ndim < 1:
    Nvec = Nvec[np.newaxis]
  Wmat = np.asarray(Wmat, dtype=np.float64)
  if Wmat.ndim < 2:
    Wmat = Wmat[np.newaxis, :]

  # Allocate and create initial guess, 
  #  which is guaranteed to have rows that sum to less than Nvec
  Nmat = np.zeros(Wmat.shape, dtype=np.int64)
  np.floor(Nvec[:,np.newaxis] * Wmat, out=Nmat)

  # Run Alg. 1 from paper
  f = Nvec[:,np.newaxis] * Wmat - Nmat
  q = (1 - f)/ Wmat
  activeRows = np.flatnonzero(Nmat.sum(axis=1) < Nvec)
  while len(activeRows) > 0:
    minIDs = np.argmin(q[activeRows], axis=1)
    ids = np.ravel_multi_index([activeRows, minIDs], Nmat.shape)
    Nmat.ravel()[ids] += 1

    stillActiveMask = Nmat[activeRows].sum(axis=1) < Nvec[activeRows]
    activeRows = activeRows[stillActiveMask]    
    ids = ids[stillActiveMask]    
    q.ravel()[ids] += 1.0 / Wmat.ravel()[ids]
  # Double-check Nmat satisfies constraints          
  assert np.all(np.abs(Nmat.sum(axis=1) - Nvec) < 0.0001)
  return Nmat

def findMode_Mult_faster(Nvec, Wmat):
  ''' Find modes to multinomial with given parameters
        vectorized to solve many problems simultaneously

      Args
      ------
      Nvec : 1D array, size P
             contains non-negative integers
      Wmat : 2D array, size P x K
             each row is valid probability vector of length K
             Wmat[p] has non-negative entries, sums to one

      Returns
      -------
      Nmat : 2D sparse array, size P x K
          Nmat[p,:] = argmax_{n} \log p_{mult}( Nmat[p] | Nvec[p], Wmat[p] ) 
  '''
  # Unpack and parse input
  Nvec = np.asarray(Nvec, dtype=np.int64)
  if Nvec.ndim < 1:
    Nvec = Nvec[np.newaxis]
  Wmat = np.asarray(Wmat, dtype=np.float64)
  if Wmat.ndim < 2:
    Wmat = Wmat[np.newaxis, :]

  # Allocate and create initial guess, 
  #  which is guaranteed to have rows that sum to less than Nvec
  Nmat = np.zeros(Wmat.shape, dtype=np.int64)
  np.floor(Nvec[:,np.newaxis] * Wmat, out=Nmat)

  # Run Alg. 1 from paper
  activeRows = np.flatnonzero(Nmat.sum(axis=1) < Nvec)

  if len(activeRows) > 0:
    f = Nvec[:,np.newaxis] * Wmat - Nmat
    q = (1 - f)/ Wmat
    q = np.take(q, activeRows, axis=0)

  while len(activeRows) > 0:
    minIDs = np.argmin(q, axis=1)
    ids = np.ravel_multi_index([activeRows, minIDs], Nmat.shape)
    Nmat.ravel()[ids] += 1

    #stillActiveMask = Nmat[activeRows].sum(axis=1) < Nvec[activeRows]
    stillActiveMask = np.take(Nmat, activeRows, axis=0).sum(axis=1) \
                       < np.take(Nvec, activeRows)
    activeRows = activeRows[stillActiveMask]    
    ids = ids[stillActiveMask]
    minIDs = minIDs[stillActiveMask]

    q = np.take(q, np.flatnonzero(stillActiveMask), axis=0)
    qids = np.ravel_multi_index([np.arange(len(minIDs)), minIDs], q.shape)
    q.ravel()[qids] += 1.0 / Wmat.ravel()[ids]
  # Double-check Nmat satisfies constraints          
  assert np.all(np.abs(Nmat.sum(axis=1) - Nvec) < 0.0001)
  return Nmat

def findMode_Mult_fastest(Nvec, Wmat):
  ''' Find modes to multinomial with given parameters
        vectorized to solve many problems simultaneously

      Args
      ------
      Nvec : 1D array, size P
             contains non-negative integers
      Wmat : 2D array, size P x K
             each row is valid probability vector of length K
             Wmat[p] has non-negative entries, sums to one

      Returns
      -------
      Nmat : 2D sparse array, size P x K
          Nmat[p,:] = argmax_{n} \log p_{mult}( Nmat[p] | Nvec[p], Wmat[p] ) 
  '''
  # Unpack and parse input
  Nvec = np.asarray(Nvec, dtype=np.int64)
  if Nvec.ndim < 1:
    Nvec = Nvec[np.newaxis]
  Wmat = np.asarray(Wmat, dtype=np.float64)
  if Wmat.ndim < 2:
    Wmat = Wmat[np.newaxis, :]

  # Allocate and create initial guess, 
  #  which is guaranteed to have rows that sum to less than Nvec
  Nmat = np.zeros(Wmat.shape, dtype=np.int64)
  Nmatorig = Nvec[:,np.newaxis] * Wmat
  np.floor(Nmatorig, out=Nmat)

  # Run Alg. 1 from paper
  activeErrors = Nvec - Nmat.sum(axis=1)
  activeRows = np.flatnonzero(activeErrors)

  if len(activeRows) > 0:
    q = 1.0 - np.take(Nmatorig, activeRows, axis=0) \
            + np.take(Nmat, activeRows, axis=0)
    q /= np.take(Wmat, activeRows, axis=0)
    activeErrors = np.take(activeErrors, activeRows)

  while len(activeRows) > 0:
    minIDs = np.argmin(q, axis=1)
    ids = np.ravel_multi_index([activeRows, minIDs], Nmat.shape)
    Nmat.ravel()[ids] += 1
    activeErrors -= 1

    stillActiveMask = activeErrors > 0
    activeRows = activeRows[stillActiveMask]   

    # Prepare for next round...
    if len(activeRows) > 0:
      activeErrors = activeErrors[stillActiveMask]
      ids = ids[stillActiveMask]
      minIDs = minIDs[stillActiveMask]

      q = np.take(q, np.flatnonzero(stillActiveMask), axis=0)
      qids = np.ravel_multi_index([np.arange(len(minIDs)), minIDs], q.shape)
      q.ravel()[qids] += 1.0 / Wmat.ravel()[ids]

  # Double-check Nmat satisfies constraints          
  assert np.all(np.abs(Nmat.sum(axis=1) - Nvec) < 0.0001)
  return Nmat


def findMode_Mult_fastest_skipsingles(Nvec, Wmat):
  ''' Find modes to multinomial with given parameters
        vectorized to solve many problems simultaneously
        tries to handle "single" case (Nvec=1) very fast

      Args
      ------
      Nvec : 1D array, size P
             contains non-negative integers
      Wmat : 2D array, size P x K
             each row is valid probability vector of length K
             Wmat[p] has non-negative entries, sums to one

      Returns
      -------
      Nmat : 2D sparse array, size P x K
          Nmat[p,:] = argmax_{n} \log p_{mult}( Nmat[p] | Nvec[p], Wmat[p] ) 
  '''
  # Unpack and parse input
  Nvec = np.asarray(Nvec, dtype=np.int64)
  if Nvec.ndim < 1:
    Nvec = Nvec[np.newaxis]
  Wmat = np.asarray(Wmat, dtype=np.float64)
  if Wmat.ndim < 2:
    Wmat = Wmat[np.newaxis, :]
  
  Nmat = toHardAssignmentMatrix(Wmat, dtype=np.int64)

  bigRows = np.flatnonzero(Nvec > 1)
  Nmatorig = np.take(Wmat, bigRows, axis=0)
  Nmatorig *= Nvec[bigRows][:, np.newaxis]
  np.floor(Nmatorig, out=np.take(Nmat, bigRows, axis=0))

  # Run Alg. 1 from paper
  activeErrors = Nvec - Nmat.sum(axis=1)
  activeMask = activeErrors > 0
  activeRows = np.flatnonzero(activeErrors)

  if len(activeRows) > 0:
    q = 1.0 - np.take(Nmatorig, np.flatnonzero(activeMask[bigRows]), axis=0) \
            + np.take(Nmat, activeRows, axis=0)
    q /= np.take(Wmat, activeRows, axis=0)
    activeErrors = np.take(activeErrors, activeRows)

  while len(activeRows) > 0:
    minIDs = np.argmin(q, axis=1)
    ids = np.ravel_multi_index([activeRows, minIDs], Nmat.shape)
    Nmat.ravel()[ids] += 1
    activeErrors -= 1

    stillActiveMask = activeErrors > 0
    activeRows = activeRows[stillActiveMask]   

    # Prepare for next round...
    if len(activeRows) > 0:
      activeErrors = activeErrors[stillActiveMask]
      ids = ids[stillActiveMask]
      minIDs = minIDs[stillActiveMask]

      q = np.take(q, np.flatnonzero(stillActiveMask), axis=0)
      qids = np.ravel_multi_index([np.arange(len(minIDs)), minIDs], q.shape)
      q.ravel()[qids] += 1.0 / Wmat.ravel()[ids]

  # Double-check Nmat satisfies constraints          
  assert np.all(np.abs(Nmat.sum(axis=1) - Nvec) < 0.0001)
  return Nmat

def findMode_Mult_sort(Nvec, Wmat):
  ''' Find modes to multinomial with given parameters
        vectorized to solve many problems simultaneously

      Args
      ------
      Nvec : 1D array, size P
             contains non-negative integers
      Wmat : 2D array, size P x K
             each row is valid probability vector of length K
             Wmat[p] has non-negative entries, sums to one

      Returns
      -------
      Nmat : 2D sparse array, size P x K
          Nmat[p,:] = argmax_{n} \log p_{mult}( Nmat[p] | Nvec[p], Wmat[p] ) 
  '''
  # Unpack and parse input
  Nvec = np.asarray(Nvec, dtype=np.int64)
  if Nvec.ndim < 1:
    Nvec = Nvec[np.newaxis]
  Wmat = np.asarray(Wmat, dtype=np.float64)
  if Wmat.ndim < 2:
    Wmat = Wmat[np.newaxis, :]

  # Allocate and create initial guess, 
  #  which is guaranteed to have rows that sum to less than Nvec
  Nmat = np.zeros(Wmat.shape, dtype=np.int64)
  np.floor(Nvec[:,np.newaxis] * Wmat, out=Nmat)

  # Run Alg. 1 from paper
  activeErrors = Nvec - Nmat.sum(axis=1)
  activeRows = np.flatnonzero(activeErrors)

  if len(activeRows) > 0:
    sortIDs = np.argsort(-1 * activeErrors)
    np.take(Wmat, sortIDs, axis=0, out=Wmat)
    np.take(Nmat, sortIDs, axis=0, out=Nmat)    
    Nvec = Nvec[sortIDs]
    activeErrors = activeErrors[sortIDs]
    activeRows = np.flatnonzero(activeErrors)

    f = Nvec[:,np.newaxis] * Wmat - Nmat
    q = (1 - f)/ Wmat
    q = np.take(q, activeRows, axis=0)
    activeErrors = np.take(activeErrors, activeRows)    
    
  while len(activeRows) > 0:
    minIDs = np.argmin(q, axis=1)
    ids = np.ravel_multi_index([activeRows, minIDs], Nmat.shape)
    Nmat.ravel()[ids] += 1
    activeErrors -= 1

    stillActiveBound = np.searchsorted(-1*activeErrors, 0, 'left')

    activeErrors = activeErrors[:stillActiveBound]
    activeRows = activeRows[:stillActiveBound]
    ids = ids[:stillActiveBound]
    minIDs = minIDs[:stillActiveBound]
    q = q[:stillActiveBound]

    qids = np.ravel_multi_index([np.arange(len(minIDs)), minIDs], q.shape)
    q.ravel()[qids] += 1.0 / Wmat.ravel()[ids]
  
  # Double-check Nmat satisfies constraints          
  assert np.all(np.abs(Nmat.sum(axis=1) - Nvec) < 0.0001)
  
  # Now just "unsort" the Nmat matrix
  unsortIDs = np.zeros(len(sortIDs), dtype=int)
  unsortIDs[sortIDs] = np.arange(len(sortIDs), dtype=int)
  np.take( Nmat, unsortIDs, axis=0, out=Nmat)
  return Nmat


def findMode_Mult_massToAdd(Nvec, Wmat):
  ''' Find modes to multinomial with given parameters
        vectorized to solve many problems simultaneously

      Args
      ------
      Nvec : 1D array, size P
             contains non-negative integers
      Wmat : 2D array, size P x K
             each row is valid probability vector of length K
             Wmat[p] has non-negative entries, sums to one

      Returns
      -------
      Nmat : 2D sparse array, size P x K
          Nmat[p,:] = argmax_{n} \log p_{mult}( Nmat[p] | Nvec[p], Wmat[p] ) 
  '''
  # Unpack and parse input
  Nvec = np.asarray(Nvec, dtype=np.int64)
  if Nvec.ndim < 1:
    Nvec = Nvec[np.newaxis]
  Wmat = np.asarray(Wmat, dtype=np.float64)
  if Wmat.ndim < 2:
    Wmat = Wmat[np.newaxis, :]

  # Allocate and create initial guess, 
  #  which is guaranteed to have rows that sum to less than Nvec
  Nmat = np.zeros(Wmat.shape, dtype=np.int64)
  np.floor(Nvec[:,np.newaxis] * Wmat, out=Nmat)

  # Run Alg. 1 from paper
  activeErrors = Nvec - Nmat.sum(axis=1)
  activeRows = np.flatnonzero(activeErrors)

  if len(activeRows) > 0:
    f = Nvec[:,np.newaxis] * Wmat - Nmat
    q = (1 - f)/ Wmat
    q = np.take(q, activeRows, axis=0)
    activeErrors = np.take(activeErrors, activeRows)    

  while len(activeRows) > 0:
    qcopy = q.copy()
    minIDs = np.argmin(q, axis=1)
    ids = np.ravel_multi_index([activeRows, minIDs], Nmat.shape)

    qids = np.ravel_multi_index([np.arange(len(minIDs)), minIDs], q.shape)
    qcopy.ravel()[qids] = np.inf
    rank2IDs = np.argmin(qcopy, axis=1)
    qids2 = np.ravel_multi_index([np.arange(len(minIDs)), rank2IDs], q.shape)

    gap = q.ravel()[qids2] - q.ravel()[qids]
    massToAdd = np.floor(gap * Wmat.ravel()[ids])
    massToAdd = np.maximum(1.0, np.minimum(massToAdd, activeErrors) )   

    activeErrors -= massToAdd
    Nmat.ravel()[ids] += massToAdd

    stillActiveMask = activeErrors > 0
    activeErrors = activeErrors[stillActiveMask]
    activeRows = activeRows[stillActiveMask]    
    ids = ids[stillActiveMask]
    minIDs = minIDs[stillActiveMask]
    massToAdd = massToAdd[stillActiveMask]

    q = np.take(q, np.flatnonzero(stillActiveMask), axis=0)
    qids = np.ravel_multi_index([np.arange(len(minIDs)), minIDs], q.shape)
    q.ravel()[qids] += massToAdd / Wmat.ravel()[ids]
  # Double-check Nmat satisfies constraints          
  assert np.all(np.abs(Nmat.sum(axis=1) - Nvec) < 0.0001)
  return Nmat

########################################################### colwisesumLogFact
###########################################################

def colwisesumLogFactorial(N):
  return colwisesumLogFactorial_fastlookup(N)

def colwisesumLogFactorial_naive(N):
  ''' Calculate column-wise sum of 

      Args
      ------
      N : 2D array, size P x K

      Returns
      --------
      H : 1D array, size K
          H[k] = np.sum( log factorial(N[:,k]))
  '''
  return np.sum(gammaln(N+1), axis=0)

def colwisesumLogFactorial_fastlookup(N):
  ''' Calculate column-wise sum of 

      Args
      ------
      N : 2D array, size P x K

      Returns
      --------
      H : 1D array, size K
          H[k] = np.sum( log factorial(N[:,k]))
  '''
  if not N.dtype == MTYPE:
    N = np.asarray(N, dtype=MTYPE)
  if not N.dtype == MTYPE:
    raise TypeError('Lookup table requires input to be type np.int64')
  return np.sum(np.take(logfactorialLookupArr, N), axis=0)

def colwisesumLogFactorial_sparselookup(N):
  ''' Calculate column-wise sum of 

      Args
      ------
      N : 2D sparse csc_matrix, size P x K

      Returns
      --------
      H : 1D array, size K
          H[k] = np.sum( log factorial(N[:,k]))
  '''
  if not N.data.dtype == MTYPE:
    raise TypeError('Lookup table requires input to be type np.int64')
  K = N.shape[1]
  H = np.zeros(K)
  for k in xrange(K):
    H[k] = np.sum( np.take(logfactorialLookupArr,
                           N.data[N.indptr[k]:N.indptr[k+1]]
                          ))
  return H

########################################################### create_lookup_table
########################################################### 
def create_lookup_table_logfactorial(Nmax):
  return gammaln(np.arange(Nmax) + 1)

########################################################### Main
########################################################### 

MTYPE = np.int32

# Create lookup table
logfactorialLookupArr = create_lookup_table_logfactorial(500)
