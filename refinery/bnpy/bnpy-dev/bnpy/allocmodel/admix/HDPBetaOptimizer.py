'''
HDPBetaOptimizer.py

Functions for obtaining point estimates of the HDP global appearance probabilities.

Model Notation
--------
HDP with a finite truncation to K active components
v    := K-length vector with entries in [0,1]
beta := K+1-length vector with entries in [0,1]
          entries must sum to unity.  sum(beta) = 1.
gamma := scalar, gamma > 0 top concentration parameter
alpha := scalar, alpha > 0 doc-level concetration parameter

Generate stick breaking fractions v 
  v[k] ~ Beta(1, gamma)
Then deterministically obtain beta
  beta[k] = v[k] prod(1 - v[:k])

Generate each document-level distribution
for d in [1, 2, ... d ... nDoc]:
  pi[d] ~ Dirichlet_{K+1}( alpha * beta )

Notes
-------
Relies on approximation to E[ log norm const of Dirichlet],
  which requires parameter gamma < 1

Set gamma close to 1 if you want variance low enough that
recovering true "beta" parameters is feasible

Set gamma close to zero (even like 0.1) makes recovered E[beta]
very different than the "true" beta
'''
import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma, polygamma
import datetime
import logging

Log = logging.getLogger('bnpy')
EPS = 10*np.finfo(float).eps

def estimate_v(gamma=1.0, alpha=0.5, nDoc=0, K=None, sumLogPi=None, initV=None, Pi=None, doVerbose=False, method='l_bfgs', **kwargs):
  ''' Solve optimization problem to estimate parameters v
      delta(v_k) = Beta( v_k | 1, gamma)

      Returns
      -------
      v : K x 1 vector of positive values
  '''
  alpha = float(alpha)
  gamma = float(gamma)
  nDoc = int(nDoc)

  if K is None:
    K = sumLogPi.size - 1
  K = int(K)

  if sumLogPi.ndim == 2:
    sumLogPi = np.sum(sumLogPi, axis=0)

  if initV is None:
    initV = np.random.rand(K)

  myFunc = lambda Cvec: objectiveFunc(Cvec, alpha, gamma, nDoc, sumLogPi)
  myGrad = lambda Cvec: objectiveGradient(Cvec, alpha, gamma, nDoc, sumLogPi)
  bestCvec, bestf, Info = scipy.optimize.fmin_l_bfgs_b(myFunc, np.log(initV), fprime=myGrad, disp=None)
  Uvec = np.exp(bestCvec)
  return Uvec

def objectiveFunc(Cvec, alpha, gamma, nDoc, sumLogPi):
  ''' Calculate unconstrained objective function for HDP variational learning
  '''
  assert not np.any(np.isnan(Cvec))
  assert not np.any(np.isinf(Cvec))

  Uvec = np.exp(Cvec)
  assert not np.any(np.isinf(Uvec))

  K = Uvec.size

  # PREPARE building-block expectations
  beta = v2beta(Uvec)

  # CALCULATE each term in the function
  obj_v = (gamma - 1) * np.sum(np.log(1-Uvec))
  obj_v -= nDoc * np.sum(gammaln(alpha * beta))
  obj_v += alpha * np.sum(beta * sumLogPi)

  assert np.all(np.logical_not(np.isnan(obj_v)))
  f = -1 * obj_v
  return f

def objectiveGradient(Cvec, alpha, gamma, nDoc, sumLogPi):
  ''' Calculate gradient of objectiveFunc, objective for HDP variational 
      Returns
      -------
        gvec : 2*K length vector,
              where each entry gives partial derivative with respect to
                  the corresponding entry of Cvec
  '''
  # UNPACK unconstrained input Cvec into intended params U

  Uvec = np.exp(Cvec)
  assert np.all(Uvec > 0)
  assert np.all(Uvec < 1)

  beta = v2beta(Uvec)
  dBdv = d_beta(Uvec, beta)

  gvecU = -1 * (gamma - 1.0) / (1.0 - Uvec)
  gvecU -= nDoc * alpha * np.dot(dBdv, digamma(alpha*beta))
  gvecU += alpha * np.dot(dBdv, sumLogPi)
  gvecU = -1 * gvecU

  # Apply chain rule!
  assert np.all(np.logical_not(np.isnan(gvecU)))
  gvecC = gvecU * Uvec

  return -1.0*gvecC

def d_beta( v, beta):
  ''' Compute gradient of beta with respect to v
      Returns
      -------
      d_beta : K x K+1 matrix, where
      d_beta[m,k] = d beta[k] / d v[m]
  '''
  K = v.size
  dbdv = np.zeros( (K, K+1) )
  for k in xrange( K ):
    dbdv[k, k] = beta[k]/v[k]
    dbdv[k, k+1:] = -1.0*beta[k+1:]/(1-v[k])
  return dbdv
  
def v2beta(v):
  ''' Convert to stick-breaking fractions v to probability vector beta
      Args
      --------
      v : K-len vector, v[k] in interval [0, 1]
      
      Returns
      --------
      beta : K+1-len vector, with positive entries that sum to 1
  '''
  v = np.asarray(v)
  c1mv = np.hstack([1.0, np.cumprod(1 - v)])
  beta = np.hstack([v,1.0]) * c1mv
  assert np.allclose(beta.sum(), 1)
  # Force away from edges 0 or 1 for numerical stability  
  beta = np.maximum(beta,EPS)
  beta = np.minimum(beta,1-EPS)
  return beta

def beta2v( beta ):
  ''' Convert probability vector beta to stick-breaking fractions v
      Args
      --------
      beta : K+1-len vector, with positive entries that sum to 1
      
      Returns
      --------
      v : K-len vector, v[k] in interval [0, 1]
  '''
  beta = np.asarray(beta)
  K = beta.size
  v = np.zeros(K-1)
  cb = beta.copy()
  for k in range(K-1):
    cb[k] = 1 - cb[k]/np.prod( cb[:k] )
    v[k] = beta[k]/np.prod( cb[:k] )
  # Force away from edges 0 or 1 for numerical stability  
  v = np.maximum(v,EPS)
  v = np.minimum(v,1-EPS)
  return v


########################################################### Test utils
###########################################################
def createToyData(v, alpha=1.0, gamma=0.5, nDoc=0, seed=42):
  ''' Generate example Pi matrix, 
        each row is a sample
  '''
  v = np.asarray(v, dtype=np.float64)
  K = v.size  
  beta = v2beta(v)

  PRNG = np.random.RandomState(seed)
  Pi = PRNG.dirichlet( gamma*beta, size=nDoc)
  return dict(Pi=Pi, alpha=alpha, gamma=gamma, nDoc=nDoc, K=K)
