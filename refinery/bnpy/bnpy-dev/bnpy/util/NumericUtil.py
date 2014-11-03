'''
NumericUtil.py

Library of efficient vectorized implementations of
 operations common to unsupervised machine learning

* inplaceExpAndNormalizeRows 
* calcRlogR
* calcRlogRdotv
* calcRlogR_allpairs
* calcRlogR_specificpairs

'''

import os
import ConfigParser
import ctypes
import numpy as np
import scipy.sparse
import timeit

import LibRlogR

def readConfigFileIntoDict(confFile, targetSecName=None):
  ''' Read contents of a config file into a dictionary

      Returns
      --------
      dict
  '''
  print confFile
  config = ConfigParser.SafeConfigParser()
  config.optionxform = str
  config.read(confFile)
  for secName in config.sections():
    if secName.count("Help") > 0:
      continue
    if targetSecName is not None:
      if secName != targetSecName:
        continue
    BigSecDict = dict(config.items(secName))
  return BigSecDict

def LoadConfig():
  global Config, cfgfilepath
  root = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
  cfgfilepath = os.path.join(root, 'config', 'numeric.platform-config')
  Config = readConfigFileIntoDict(cfgfilepath, 'LibraryPrefs')


########################################################### inplace exp
###########################################################
def inplaceExp(R):
  ''' Calculate exp of each entry of input matrix, done in-place.
  '''
  if Config['inplaceExpAndNormalizeRows'] == "numexpr" and hasNumexpr:
    return inplaceExp_numexpr(R)
  else:  
    return inplaceExp_numpy(R)

def inplaceExp_numpy(R):
  ''' Calculate exp of each entry of input matrix, done in-place.
  '''
  np.exp(R, out=R)

def inplaceExp_numexpr(R):
  ''' Calculate exp of each entry of input matrix, done in-place.
  '''
  ne.evaluate("exp(R)", out=R)

########################################################### exp and normalize
###########################################################
def inplaceExpAndNormalizeRows(R):
  ''' Calculate exp in numerically stable way (first subtract max),
       and normalize the rows, all done in-place on the input matrix.
  '''
  if Config['inplaceExpAndNormalizeRows'] == "numexpr" and hasNumexpr:
    return inplaceExpAndNormalizeRows_numexpr(R)
  else:  
    return inplaceExpAndNormalizeRows_numpy(R)

def inplaceExpAndNormalizeRows_numpy(R):
  ''' Calculate exp in numerically stable way (first subtract max),
       and normalize the rows, all done in-place on the input matrix.
  '''
  R -= np.max(R, axis=1)[:,np.newaxis]
  np.exp(R, out=R)
  R /= R.sum(axis=1)[:,np.newaxis]

def inplaceExpAndNormalizeRows_numexpr(R):
  ''' Calculate exp in numerically stable way (first subtract max),
       and normalize the rows, all done in-place on the input matrix.
  '''
  R -= np.max(R, axis=1)[:,np.newaxis]
  ne.evaluate("exp(R)", out=R)
  R /= R.sum(axis=1)[:,np.newaxis]


########################################################### standard RlogR
###########################################################
def calcRlogR(R):
  ''' Calculate sum across columns of element-wise product R*log(R),
        for NxD array R
      Uses faster numexpr library if available, but safely falls back
        to plain numpy otherwise.      

      Returns
      --------
      S : K-len vector where S[k] = \sum_{n=1}^N r[n,k] log r[n,k]
  '''
  if Config['calcRlogR'] == "numexpr" and hasNumexpr:
    return calcRlogR_numexpr(R)
  else:  
    return calcRlogR_numpy(R)

def calcRlogR_numpy(R):
  return np.sum(R * np.log(R), axis=0)

def calcRlogR_numexpr(R):
  return ne.evaluate("sum(R*log(R), axis=0)")

########################################################### standard RlogRdotv
###########################################################
def calcRlogRdotv(R, v):
  ''' Calculate dot product of (R * log R) and v,
        for NxD array R and vector v
      Uses faster numexpr library if available, but safely falls back
        to plain numpy otherwise.      

      Returns
      --------
      S : K-len vector where S[k] = \sum_{n=1}^N v[n] (r[n,k] log r[n,k])
  '''
  if Config['calcRlogRdotv'] == "numexpr" and hasNumexpr:
    return calcRlogRdotv_numexpr(R, v)
  else:  
    return calcRlogRdotv_numpy(R, v)

def calcRlogRdotv_numpy(R, v):
  return np.dot( v, R * np.log(R))

def calcRlogRdotv_numexpr(R, v):
  RlogR = ne.evaluate("R*log(R)")
  return np.dot(v, RlogR)

########################################################### all-pairs RlogR
###########################################################
def calcRlogR_allpairs(R):
  ''' Calculate column sums of element-wise product Rm*log(Rm)
        where Rm represents all pair-wise merges of columns of R.

      Uses faster numexpr library if available, but safely falls back
        to plain numpy otherwise.      

      Returns
      --------
      Z : KxK matrix, where Z[a,b] = sum(Rab*log(Rab)), Rab = R[:,a] + R[:,b]
          only upper-diagonal entries of Z are non-zero,
          since we restrict potential pairs a,b to satisfy a < b
  '''
  if Config['calcRlogR'] == "numexpr" and hasNumexpr:
    return calcRlogR_allpairs_numexpr(R)
  else:  
    return calcRlogR_allpairs_numpy(R)

def calcRlogR_allpairs_numpy(R):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in xrange(K-1):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    curR *= np.log(curR)
    Z[jj,jj+1:] = np.sum(curR, axis=0)
  return Z

def calcRlogR_allpairs_numexpr(R):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in xrange(K-1):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    curZ = ne.evaluate("sum(curR * log(curR), axis=0)")
    Z[jj,jj+1:] = curZ
  return Z

def calcRlogR_allpairs_c(R):
  return LibRlogR.calcRlogR_allpairs_c(R)

########################################################### all-pairs
###########################################################  RlogRdotv
def calcRlogRdotv_allpairs(R, v):
  ''' Calculate dot product dot(v, Rm * log(Rm)),
        where Rm represents all pair-wise merges of columns of R

      Uses faster numexpr library if available, but safely falls back
        to plain numpy otherwise.      

      Args
      --------
      R : NxK matrix
      v : N-vector

      Returns
      --------
      Z : KxK matrix, where Z[a,b] = v dot (Rab*log(Rab)), Rab = R[:,a] + R[:,b]
          only upper-diagonal entries of Z are non-zero,
          since we restrict potential pairs a,b to satisfy a < b
  '''
  if Config['calcRlogRdotv'] == "numexpr" and hasNumexpr:
    return calcRlogRdotv_allpairs_numexpr(R, v)
  else:  
    return calcRlogRdotv_allpairs_numpy(R, v)

def calcRlogRdotv_allpairs_numpy(R, v):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in range(K):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    curR *= np.log(curR)
    Z[jj,jj+1:] = np.dot(v,curR)
  return Z

def calcRlogRdotv_allpairs_numexpr(R, v):
  K = R.shape[1]
  Z = np.zeros((K,K))
  for jj in xrange(K-1):
    curR = R[:,jj][:,np.newaxis] + R[:, jj+1:]
    ne.evaluate("curR * log(curR)", out=curR)
    curZ = np.dot(v, curR)
    Z[jj,jj+1:] = curZ
  return Z

def calcRlogRdotv_allpairs_c(R, v):
  return LibRlogR.calcRlogRdotv_allpairs_c(R, v)

########################################################### specific-pairs
###########################################################  RlogRdotv
def calcRlogRdotv_specificpairs(R, v, mPairs):
  ''' Calculate dot product dot(v, Rm * log(Rm)),
        where Rm represents specific pair-wise merges of columns of R

      Uses faster numexpr library if available, but safely falls back
        to plain numpy otherwise.      

      Args
      --------
      R : NxK matrix
      v : N-vector
      mPairs : list of possible merge pairs, where each pair is a tuple
                 [(a,b), (c,d), (e,f)]

      Returns
      --------
      Z : KxK matrix, where Z[a,b] = v dot (Rab*log(Rab)), Rab = R[:,a] + R[:,b]
          only upper-diagonal entries of Z specified by mPairs are non-zero,
          since we restrict potential pairs a,b to satisfy a < b
  '''
  if Config['calcRlogRdotv'] == "numexpr" and hasNumexpr:
    return calcRlogRdotv_specificpairs_numexpr(R, v, mPairs)
  else:  
    return calcRlogRdotv_specificpairs_numpy(R, v, mPairs)

def calcRlogRdotv_specificpairs_numpy(R, v, mPairs):
  K = R.shape[1]
  ElogqZMat = np.zeros((K, K))
  if K == 1:
    return ElogqZMat
  for (kA, kB) in mPairs:
    curWV = R[:,kA] + R[:, kB]
    curWV *= np.log(curWV)
    ElogqZMat[kA,kB] = np.dot(v, curWV)
  return ElogqZMat

def calcRlogRdotv_specificpairs_numexpr(R, v, mPairs):
  K = R.shape[1]
  ElogqZMat = np.zeros((K, K))
  if K == 1:
    return ElogqZMat
  for (kA, kB) in mPairs:
    curR = R[:,kA] + R[:, kB]
    ne.evaluate("curR * log(curR)", out=curR)
    ElogqZMat[kA,kB] = np.dot(v, curR)
  return ElogqZMat

def calcRlogRdotv_specificpairs_numpyvec(R, v, mPairs):
  ''' Attempt to speed up by handling all partners of comp kA at once.
      Surprisingly seems to be much slower. Forget it.
  '''
  from collections import defaultdict
  PartnerDict = defaultdict(lambda: list())
  for kA, kB in mPairs:
    PartnerDict[kA].append(kB)
  K = R.shape[1]
  ElogqZMat = np.zeros((K, K))
  if K == 1:
    return ElogqZMat
  for kA in PartnerDict:
    curWV = R[:,kA][:,np.newaxis] + R[:, PartnerDict[kA]]
    curWV *= np.log(curWV)
    ElogqZMat[kA,PartnerDict[kA]] = np.dot(v, curWV)
  return ElogqZMat


def calcRlogRdotv_specificpairs_c(R, v, mPairs):
  return LibRlogR.calcRlogRdotv_specificpairs_c(R, v, mPairs)

########################################################### AutoConfigure
###########################################################
def autoconfigure():
  ''' Perform timing experiments on current hardware to assess which
       of various implementations is the fastest for each key routine
  '''
  config = ConfigParser.SafeConfigParser()
  config.optionxform = str
  config.read(cfgfilepath)
  methodNames = ['inplaceExpAndNormalizeRows', 'calcRlogR', 'calcRlogRdotv']
  for mName in methodNames:
    if mName == 'inplaceExpAndNormalizeRows':
      expectedGainFactor = runTimingExperiment_inplaceExpAndNormalizeRows()
    elif mName == 'calcRlogR':
      expectedGainFactor = runTimingExperiment_calcRlogR()
    elif mName == 'calcRlogRdotv':
      expectedGainFactor = runTimingExperiment_calcRlogRdotv()
    print mName,
    if expectedGainFactor > 1.05:
      config.set('LibraryPrefs', mName, 'numexpr')
      print "numexpr preferred: %.2f X faster" % (expectedGainFactor)
    else:
      config.set('LibraryPrefs', mName, 'numpy')
      print "numpy preferred: %.2f X faster" % (expectedGainFactor)
    with open(cfgfilepath,'w') as f:
      config.write(f)
  LoadConfig()

def runTimingExperiment_inplaceExpAndNormalizeRows(N=2e5, D=100, repeat=3):
  if not hasNumexpr:
    return 0
  setup = "import numpy as np; import numexpr as ne;"
  setup += "from bnpy.util import NumericUtil as N;"
  setup += "R = np.random.rand(%d, %d)" % (N, D)
  elapsedTimes_np = timeit.repeat("N.inplaceExpAndNormalizeRows_numpy(R)", 
                    setup=setup, number=1, repeat=repeat)
  elapsedTimes_ne = timeit.repeat("N.inplaceExpAndNormalizeRows_numexpr(R)",
                    setup=setup, number=1, repeat=repeat)
  meanTime_np = np.mean(elapsedTimes_np)
  meanTime_ne = np.mean(elapsedTimes_ne)
  expectedGainFactor = meanTime_np / meanTime_ne
  return expectedGainFactor

def runTimingExperiment_calcRlogR(N=2e5, D=100, repeat=3):
  if not hasNumexpr:
    return 0
  setup = "import numpy as np; import numexpr as ne;"
  setup += "import bnpy.util.NumericUtil as N;"
  setup += "R = np.random.rand(%d, %d)" % (N, D)
  elapsedTimes_np = timeit.repeat("N.calcRlogR_numpy(R)", 
                    setup=setup, number=1, repeat=repeat)
  elapsedTimes_ne = timeit.repeat("N.calcRlogR_numexpr(R)",
                    setup=setup, number=1, repeat=repeat)
  meanTime_np = np.mean(elapsedTimes_np)
  meanTime_ne = np.mean(elapsedTimes_ne)
  expectedGainFactor = meanTime_np / meanTime_ne
  return expectedGainFactor

def runTimingExperiment_calcRlogRdotv(N=2e5, D=100, repeat=3):
  if not hasNumexpr:
    return 0
  setup = "import numpy as np; import numexpr as ne;"
  setup += "import bnpy.util.NumericUtil as N;"
  setup += "R = np.random.rand(%d, %d);" % (N, D)
  setup += "v = np.random.rand(%d)" % (N)
  elapsedTimes_np = timeit.repeat("N.calcRlogRdotv_numpy(R, v)", 
                    setup=setup, number=1, repeat=repeat)
  elapsedTimes_ne = timeit.repeat("N.calcRlogRdotv_numexpr(R, v)",
                    setup=setup, number=1, repeat=repeat)
  meanTime_np = np.mean(elapsedTimes_np)
  meanTime_ne = np.mean(elapsedTimes_ne)
  expectedGainFactor = meanTime_np / meanTime_ne
  return expectedGainFactor

########################################################### MAIN
###########################################################

hasNumexpr = True
try:
  import numexpr as ne
  if 'OMP_NUM_THREADS' in os.environ:
    ne.set_num_threads(os.environ['OMP_NUM_THREADS'])
except:
  hasNumexpr = False

LoadConfig()

