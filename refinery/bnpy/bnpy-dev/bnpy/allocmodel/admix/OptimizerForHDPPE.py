'''
OptimizerForHDPPE.py

Model Notation
--------
Dirichlet-Multinomial model with K+1 possible components

v    := K-length vector with entries in [0,1]
beta := K+1-length vector with entries in [0,1]
          entries must sum to unity.  sum(beta) = 1.
alpha0 := scalar, alpha0 > 0

Generate stick breaking fractions v 
  v[k] ~ Beta(1, alpha0)
Then deterministically obtain beta
  beta[k] = v[k] prod(1 - v[:k]), k = 1, 2, ... K
  beta[K+1] = prod_k=1^K 1-v[k]
Then draw observed probability vectors
  pi[d] ~ Dirichlet(gamma * beta), for d = 1, 2, ... D

CONSTRAINED Optimization Problem
----------
v* = argmax_v  log p(pi | v) + log p( v ), subject to 0 <= v <= 1

UNCONSTRAINED Problem
----------
c* = argmax_c log p(pi | v) + log p (v), where v = sigmoid(c),  -Inf < c < Inf
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

def estimate_v(sumLogPi=None, nDoc=0, gamma=1.0, alpha0=1.0, initv=None, approx_grad=False, **kwargs):
  ''' Run gradient optimization to estimate best v for specified problem

      Returns
      -------- 
      vhat : K-vector of values, 0 < v < 1
      fofvhat: objective function value at vhat
      Info : dict with info about estimation algorithm
  '''
  sumLogPi = np.squeeze(np.asarray(sumLogPi, dtype=np.float64))
  assert sumLogPi.ndim == 1
  K = sumLogPi.size - 1
  if initv is None:
    initv = 1.0/(1.0+alpha0) * np.ones(K)
  assert initv.size == K

  initc = invsigmoid(initv)
  myFunc = lambda c: objFunc_c(c, sumLogPi, nDoc, gamma, alpha0)
  myGrad = lambda c: objGrad_c(c, sumLogPi, nDoc, gamma, alpha0)
  
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      chat, fhat, Info = scipy.optimize.fmin_l_bfgs_b(myFunc, initc,
                                                  approx_grad=approx_grad,
                                                  fprime=myGrad, disp=None,
                                                  factr=1e10,
                                                  **kwargs)
    except RuntimeWarning:
      Info = dict(warnflag=2, task='Overflow!')
      chat = initc
      fhat = myFunc(chat)

  if Info['warnflag'] > 1:
    print "******", Info['task']
    raise ValueError("Optimization failed")

  vhat = sigmoid(chat)
  Info['initv'] = initv
  Info['objFunc'] = lambda v: objFunc_v(v, sumLogPi, nDoc, gamma, alpha0)
  Info['gradFunc'] = lambda v: objGrad_v(v, sumLogPi, nDoc, gamma, alpha0)
  return vhat, fhat, Info

########################################################### Objective/gradient
########################################################### in terms of v

def objFunc_v(v, sumLogPi, nDoc, gamma, alpha0):
  ''' Returns scalar value of constrained objective function
      Args
      -------
      v := K-vector of real numbers, subject to 0 < v < 1

      Returns
      -------
      f := -1 * L(v), where L is ELBO objective function (log posterior prob)
  '''
  # log prior
  logpV = (alpha0 - 1) * np.sum(np.log(1.-v))
  # log likelihood
  beta = v2beta(v)
  logpPi_const = gammaln(gamma) - np.sum(gammaln(gamma*beta))
  logpPi = np.inner(gamma*beta - 1, sumLogPi)
  return -1.0 * (nDoc*logpPi_const + logpPi + logpV)

def objGrad_v(v, sumLogPi, nDoc, gamma, alpha0):
  ''' Returns K-vector gradient of the constrained objective
      Args
      -------
      v := K-vector of real numbers, subject to 0 < v < 1

      Returns
      -------
      g := K-vector of real numbers, 
            g[k] = gradient of -1 * L(v) with respect to v[k]
  '''
  K = v.size
  beta = v2beta(v)
  dv_logpV = (1 - alpha0) / (1-v)

  dv_logpPi_const = np.zeros(K)
  psibeta = digamma(gamma*beta) * beta
  for k in xrange(K):
    Sk = -1.0*psibeta[k]/v[k] + np.sum( psibeta[k+1:]/(1-v[k]) )
    dv_logpPi_const[k] = nDoc * gamma * Sk

  dv_logpPi = np.zeros(K)
  sbeta = sumLogPi * beta
  for k in xrange(K):
    Sk = sbeta[k]/v[k] - np.sum( sbeta[k+1:]/(1-v[k]) )
    dv_logpPi[k] = gamma * Sk

  return -1.0* ( dv_logpV + dv_logpPi_const + dv_logpPi )


def objGrad_v_FAST(v, sumLogPi, nDoc, gamma, alpha0):
  ''' Returns K-vector gradient of the constrained objective
      Args
      -------
      v := K-vector of real numbers, subject to 0 < v < 1

      Returns
      -------
      g := K-vector of real numbers, 
            g[k] = gradient of -1 * L(v) with respect to v[k]
  '''
  K = v.size
  beta = v2beta(v)
  dv_logpV = (1 - alpha0) / (1-v)

  diagIDs = np.diag_indices(K)
  lowTriIDs = np.tril_indices(K, -1)
  S = np.tile( sumLogPi * beta, (K,1))
  S /= (1.0 - v[:, np.newaxis])
  S[diagIDs] *= -1 * (1.0 - v)/v
  S[lowTriIDs] = 0
  dv_logpPi = gamma * np.sum(S, axis=1)

  S = np.tile( digamma(gamma*beta) * beta, (K,1))
  S /= (1.0 - v[:, np.newaxis])
  S[diagIDs] *= -1 * (1.0 - v)/v
  S[lowTriIDs] = 0
  dv_logpPi_const = nDoc * gamma * np.sum(S, axis=1)

  return -1.0* ( dv_logpV + dv_logpPi_const + dv_logpPi )


########################################################### Objective/gradient
########################################################### in terms of c

def objFunc_c(c, *args):
  ''' Returns scalar value of unconstrained objective function
      Args
      -------
      c := K-vector of real numbers

      Returns
      -------
      f := -1 * L( v2c(c) ), where L is ELBO objective (log posterior)
  '''
  v = sigmoid(c)
  # Force away from edges 0 or 1 for numerical stability  
  #v = np.maximum(v,EPS)
  #v = np.minimum(v,1.0-EPS)

  return objFunc_v(v, *args)

def objGrad_c(c, *args):
  ''' Returns K-vector gradient of unconstrained objective function
      Args
      -------
      c := K-vector of real numbers

      Returns
      -------
      g := K-vector of real numbers, 
            g[k] = gradient of -1 * L( v2c(c) ) with respect to c[k]
  '''
  v = sigmoid(c)
  # Force away from edges 0 or 1 for numerical stability  
  #v = np.maximum(v,EPS)
  #v = np.minimum(v,1.0-EPS)

  dfdv = objGrad_v(v, *args)
  dvdc = v * (1-v)
  dfdc = dfdv * dvdc
  return dfdc

########################################################### Transform funcs
########################################################### v2c, c2v

def sigmoid(c):
  ''' sigmoid(c) = 1./(1+exp(-c))
  '''
  return 1.0/(1.0 + np.exp(-c))

def invsigmoid(v):
  ''' Returns the inverse of the sigmoid function
      v = sigmoid(invsigmoid(v))

      Args
      --------
      v : positive vector with entries 0 < v < 1
  '''
  assert np.all( v <= 1-EPS)
  assert np.all( v >= EPS)
  return -np.log((1.0/v - 1))

########################################################### Transform funcs
########################################################### v2beta, beta2v

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
  beta = np.hstack([1.0, np.cumprod(1-v)])
  beta[:-1] *= v
  # Force away from edges 0 or 1 for numerical stability  
  #beta = np.maximum(beta,EPS)
  #beta = np.minimum(beta,1-EPS)
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