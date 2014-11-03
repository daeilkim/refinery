import numpy as np

def VectorProxFunc(xa, xb, atol=0.1, rtol=0.15):
  return np.abs(xa - xb) <= atol * (np.abs(xa)< atol) \
                             + rtol * np.maximum(np.abs(xa), np.abs(xb))

def CovMatProxFunc(Sa, Sb, rtol=0.1):
  if Sa.ndim == 3:
    mask = np.zeros(Sa.shape[0])
    for k in range(Sa.shape[0]):
      mask[k] = np.all(CovMatProxFunc(Sa[k], Sb[k]))
    return mask
  else:
    assert Sa.ndim == 2
    da = np.diag(Sa)
    db = np.diag(Sb)
    diagMask = np.abs(da-db) <= rtol * np.abs(da)

    # For off-diagonal entries, need absolute tolerance for entry i,j
    #  that scales with the *larger* of the two variances involved sig_i, sig_j
    D = da.size
    oH = np.tile( da[:,np.newaxis], D)
    oV = oH.T
    offdiag_atol = 0.9 * np.maximum( oH, oV )
    mask = np.abs(Sa - Sb) <= (offdiag_atol + rtol * np.abs(Sa))
    mask[np.diag_indices(D)] = diagMask 
    return mask

def ProbVectorProxFunc(p, q, atol=0.01):
  return np.abs(p - q) <= atol


def pprint( arr, msg='', fmt='% 9.3f', replaceVal=None, replaceText='-', rstart=0, cstart=0, Kmax=7):
  ''' Pretty print array
  '''
  arr = np.asarray(arr)
  if arr.ndim == 0:
    s = fmt % (arr)
  elif arr.ndim == 1:
    s = ' '.join( [fmt % (x) for x in arr[rstart:rstart+Kmax]])
    if len(msg) > 0:
      s += ' ' + msg
  elif arr.ndim == 2:
    s = ''
    for r in xrange(rstart, rstart+np.minimum(Kmax,arr.shape[0])):
      s += ' '.join( [fmt % (x) for x in arr[r, cstart:cstart+Kmax]])
      if r == rstart and len(msg) > 0:
        s += ' ' + msg
      s += '\n'
  else:
    s = str(arr)
  if replaceVal is not None:
    key = fmt % (replaceVal)
    keyfmt = '%' + str(len(key)) + 's'
    s = s.replace( key, keyfmt % (replaceText))
  print s

def buildArrForObsModelParams(compList, key):
  Xlist = list()
  for comp in compList:
    Xlist.append(getattr(comp,key))
  if Xlist[0].ndim == 1:
    return np.vstack(Xlist)
  else:  
    X = np.zeros( (len(Xlist), Xlist[0].shape[0], Xlist[0].shape[1]))
    for k in range(len(Xlist)):
      X[k] = Xlist[k]
    return X
  
Cache = dict()
def MakeZMGaussData(Sigma, Nk, seed=1234):
  if seed in Cache:
    return Cache[seed]
  PRNG = np.random.RandomState(seed)
  if Sigma.ndim == 3:
    K = Sigma.shape[0]
  else:
    K = 1
    Sigma = Sigma[np.newaxis,:]
  Xlist = list()
  for k in range(K):
    Xk = PRNG.multivariate_normal(np.zeros(Sigma.shape[-1]), Sigma[k], Nk)
    Xlist.append(Xk)
  X = np.vstack(Xlist)
  Cache[seed] = X
  return X


Cache = dict()
def MakeGaussData(Mu, Sigma, Nk, seed=1234):
  if seed in Cache:
    return Cache[seed]
  PRNG = np.random.RandomState(seed)
  if Sigma.ndim == 3:
    K = Sigma.shape[0]
  else:
    K = 1
    Sigma = Sigma[np.newaxis,:]
  Xlist = list()
  for k in range(K):
    Xk = PRNG.multivariate_normal(Mu[k], Sigma[k], Nk)
    Xlist.append(Xk)
  X = np.vstack(Xlist)
  Cache[seed] = X
  return X
