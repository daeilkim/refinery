'''
HMMUtil.py

Provides standard message-passing algorithms for inference in HMMs,
  such as the forward-backward algorithm

Intentionally separated from rest of HMM code, so that we can swap in 
  any fast routine for this calculation with ease.
'''
import numpy as np

def FwdBwdAlg(PiInit, PiMat, logSoftEv):
  '''Execute forward-backward message passing algorithm
       given HMM state transition params and log likelihoods of each observation

     Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 2D array, size KxK
            piMat[j] is transition distribution from state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     logSoftEv : 2D array, size TxK
            logSoftEv[t] := log p( x[t] | z[tk] = 1)
                         log likelihood of observation t under state k
                         if given exactly, 
                           * resp, respPair will be exact
                           * logMargPrSeq will be exact
                         if given up to an additive constant,
                           * resp, respPair will be exact
                           * logMargPrSeq will be off by an additive constant
     Returns
     -------
     resp : 2D array, size T x K
            resp[t,k] = marg. prob. that step t assigned to state K
                        p( z[t,k] = 1 | x[1], x[2], ... x[T])
     respPair : 2D array, size T x K x K
            respPair[t,j,k] = marg. prob. that both
                              * step t-1 assigned to state j
                              * step t assigned to state k
                        p( z[t-1,j] = 1, z[t,k] = 1 | x[1], x[2], ... x[T])
            respPair[0,:,:] is undefined, but kept to match indexing consistent.

     logMargPrSeq : scalar real
            logMargPrSeq = joint log probability of the observed sequence
                        log p( x[1], x[2], ... x[T] )  
  '''
  PiInit, PiMat, K = _parseInput_TransParams(PiInit, PiMat)
  logSoftEv = _parseInput_SoftEv(logSoftEv, K)
  T = logSoftEv.shape[0]

  SoftEv, lognormC = expLogLik(logSoftEv)
  
  fmsg, margPrObs = FwdAlg(PiInit, PiMat, SoftEv )
  bmsg = BwdAlg( PiInit, PiMat, SoftEv, margPrObs)
  
  respPair = np.zeros( (T,K,K) )
  for t in xrange( 1, T ):
    respPair[t] = PiMat * np.outer(fmsg[t-1], bmsg[t] * SoftEv[t]) / margPrObs[t]

    assert np.allclose(respPair[t].sum(), 1.0)
  logMargPrSeq = np.log(margPrObs).sum() + lognormC.sum()

  resp = fmsg * bmsg
  return resp, respPair, logMargPrSeq


def FwdAlg(PiInit, PiMat, SoftEv):
  ''' Execute forward message-passing on an observed sequence
       given HMM state transition params and likelihoods of each observation

     Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 2D array, size KxK
            piMat[j] is transition distribution from state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     SoftEv : 2D array, size TxK
            SoftEv[t] := p( x[t] | z[tk] = 1)
                         likelihood of observation t under state k
                         given up to an additive constant for each t
     Returns
     -------
        fmsg : 2D array, size T x K
                  fmsg[t,k] = p( z[t,k] = 1 | x[1] ... x[t] )
        margPrObs : 1D array, size T
                  margPrObs[t] = p( x[t] | x[1], x[2], ... x[t-1] )
  '''
  T = SoftEv.shape[0]
  K = PiInit.size
  PiTMat = PiMat.T

  fmsg = np.empty( (T,K) )
  margPrObs = np.zeros( T )
  for t in xrange( 0, T ):
    if t == 0:
      fmsg[t] = PiInit * SoftEv[0]
    else:
      fmsg[t] = np.dot(PiTMat, fmsg[t-1]) * SoftEv[t]
    margPrObs[t] = np.sum( fmsg[t] )
    fmsg[t] /= margPrObs[t]
  return fmsg, margPrObs
  
def BwdAlg(PiInit, PiMat, SoftEv, margPrObs):
  '''Execute backward message-passing on an observed sequence
       given HMM state transition params and likelihoods of each observation
     Requires running forward filtering first, to obtain correct scaling.

     Args
     -------
     piInit : 1D array, size K
            initial transition distribution to each of the K states
            must be valid probability vector (positive entries, sums to one)
     piMat  : 2D array, size KxK
            piMat[j] is transition distribution from state j to all K states.
            each row must be probability vector (positive entries, sums to one)
     SoftEv : 2D array, size TxK
            SoftEv[t] := p( x[t] | z[tk] = 1)
                         likelihood of observation t under state k
                         given up to an additive constant for each t
     margPrObs : 1D array, size T
            margPrObs[t] := p( x[t] | x[1], x[2], ... x[t-1] )
            this is returned by FwdAlg

     Returns
     -------
     bmsg : 2D array, size TxK
              bmsg[t,k] = p( x[t+1], x[t+2], ... x[T] |  z[t,k] = 1 )
                          -------------------------------------
                          p( x[t+1], x[t+2], ... x[T] |  x[1] ... x[t])
  '''
  T = SoftEv.shape[0]
  K = PiInit.size
  bmsg = np.ones( (T,K) )
  for t in xrange( T-2, -1, -1 ):
    bmsg[t] = np.dot(PiMat, bmsg[t+1] * SoftEv[t+1] )
    bmsg[t] /= margPrObs[t+1]
  return bmsg

########################################################### expLogLik
###########################################################
def expLogLik(logSoftEv, axis=1):
  ''' Return element-wise exp of input log likelihood
        guaranteed not to underflow
    
      Returns
      --------
      SoftEv : 2D array, size TxK
                equal to exp(logSoftEv), up to prop constant for each row
      lognormC : 1D array, size T
                gives log of the prop constant for each row
  '''
  lognormC = np.max(logSoftEv, axis)
  if axis==0:
    logSoftEv = logSoftEv - lognormC[np.newaxis,:]
  elif axis==1:
    logSoftEv = logSoftEv - lognormC[:,np.newaxis]
  SoftEv = np.exp(logSoftEv)
  return SoftEv, lognormC

########################################################### Parse input
###########################################################
def _parseInput_TransParams(PiInit, PiMat):
  PiInit = np.asarray(PiInit, dtype=np.float64)
  PiMat = np.asarray(PiMat, dtype=np.float64)
  assert PiInit.ndim == 1
  K0 = PiInit.shape[0]
  assert PiMat.ndim == 2
  J, K = PiMat.shape
  assert J == K
  assert K0 == K
  return PiInit, PiMat, K

def _parseInput_SoftEv(logSoftEv, K):
  logSoftEv = np.asarray(logSoftEv, dtype=np.float64)
  Tl, Kl = logSoftEv.shape
  assert Kl == K
  return logSoftEv
