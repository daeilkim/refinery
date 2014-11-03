'''
OptimizerForHDPFullVarModel.py

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
q(v) = Beta( v | u1, u0)
u* = argmax_u  E_q[log p(pi | v) + log p( v ) - log q(v) ], subject to 0 <= u

UNCONSTRAINED Problem
----------
c* = argmax_c E_q[log p(pi | v) + log p( v ) - log q(v) ], u = exp(c), c is real
'''

import warnings
import numpy as np
import scipy.optimize
import scipy.io
from scipy.special import gammaln, digamma, polygamma
import datetime
import logging
import itertools

Log = logging.getLogger('bnpy')
EPS = 10*np.finfo(float).eps

lowTriIDsDict = dict()
def get_lowTriIDs(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K, -1)
    lowTriIDsDict[K] = ltIDs
    return ltIDs

def get_lowTriIDs_flat(K):
  if K in lowTriIDsDict:
    return lowTriIDsDict[K]
  else:
    ltIDs = np.tril_indices(K, -1)
    lowTriIDsDict[K] = np.ravel_multi_index(ltIDs, (K,K))
    return ltIDs

def estimate_u_multiple_tries(sumLogPi=None, nDoc=0, gamma=1.0, alpha0=1.0,
                              initu=None, initU=None, approx_grad=False,
                              fList=[1e7, 1e8, 1e10], **kwargs):
  ''' Estimate 2K-vector "u" via gradient descent,
        gracefully using multiple restarts with progressively weaker tolerances
        until one succeeds

      Returns
      --------
      u : 2K-vector of positive parameters
      fofu : scalar value of minimization objective
      Info : dict

      Raises
      --------
      ValueError with FAILURE in message if all restarts fail
  '''
  K = sumLogPi.size - 1
  if initU is not None:
    initu = initU
  if initu is not None and not np.allclose(initu[-K:], alpha0):
    uList = [initu, None]
  else:
    uList = [None] 

  nOverflow = 0
  u = None
  Info = dict()
  msg = ''
  for trial, myTuple in enumerate(itertools.product(uList, fList)):
    initu, factr = myTuple
    try:
      u, fofu, Info = estimate_u(sumLogPi, nDoc, gamma, alpha0,
                                initu=initu, factr=factr, approx_grad=approx_grad)
      Info['nRestarts'] = trial
      Info['factr'] = factr
      Info['msg'] = Info['task']
      del Info['grad']
      del Info['task']
      break
    except ValueError as err:
      if str(err).count('FAILURE') == 0:
        raise err
      msg = str(err)
      if str(err).count('overflow') > 0:
        nOverflow += 1

  if u is None:
    raise ValueError("FAILURE! " + msg)
  Info['nOverflow'] = nOverflow
  return u, fofu, Info      

def estimate_u(sumLogPi=None, nDoc=0, gamma=1.0, alpha0=1.0, initu=None, approx_grad=False, factr=1.0e7, **kwargs):
  ''' Run gradient optimization to estimate best v for specified problem

      Returns
      -------- 
      vhat : K-vector of values, 0 < v < 1
      fofvhat: objective function value at vhat
      Info : dict with info about estimation algorithm

      Raises
      --------
      ValueError on an overflow, any detection of NaN, or failure to converge
  '''
  sumLogPi = np.squeeze(np.asarray(sumLogPi, dtype=np.float64))
  assert sumLogPi.ndim == 1
  K = sumLogPi.size - 1
  if initu is None:
    initu = np.hstack( [np.ones(K), alpha0*np.ones(K)])
  assert initu.size == 2*K

  initc = np.log(initu)
  myFunc = lambda c: objFunc_c(c, sumLogPi, nDoc, gamma, alpha0)
  myGrad = lambda c: objGrad_c(c, sumLogPi, nDoc, gamma, alpha0)
  
  with warnings.catch_warnings():
    warnings.filterwarnings('error', category=RuntimeWarning,
                               message='overflow')
    try:
      chat, fhat, Info = scipy.optimize.fmin_l_bfgs_b(myFunc, initc,
                                                  fprime=myGrad, disp=None,
                                                  approx_grad=approx_grad,
                                                  factr=factr,
                                                  **kwargs)
    except RuntimeWarning:
      raise ValueError("FAILURE: overflow!" )
    except AssertionError:
      raise ValueError("FAILURE: NaN found!")
      
  if Info['warnflag'] > 1:
    raise ValueError("FAILURE: " + Info['task'])

  uhat = np.exp(chat)
  Info['initu'] = initu
  Info['objFunc'] = lambda u: objFunc_u(u, sumLogPi, nDoc, gamma, alpha0)
  Info['gradFunc'] = lambda u: objGrad_u(u, sumLogPi, nDoc, gamma, alpha0)
  return uhat, fhat, Info

########################################################### Objective/gradient
########################################################### in terms of u
def objFunc_u(u, sumLogPi, nDoc, gamma, alpha0):
  ''' Returns scalar value of constrained objective function
      Args
      -------
      u := 2K-vector of real numbers, subject to 0 < u

      Returns
      -------
      f := -1 * L(u), where L is ELBO objective function (log posterior prob)
  '''
  assert not np.any(np.isnan(u))

  # PREPARE building-block expectations
  u1, u0 = _unpack(u)
  E = _calcExpectations(u1, u0)

  # CALCULATE each term in the function
  K = u1.size
  kvec = K+1 - np.arange(1, K+1)

  Elog_pmq_v = np.sum(gammaln(u1) + gammaln(u0) - gammaln(u1 + u0)) \
                + np.inner(1.0 - u1, E['logv']) \
                + np.inner(alpha0 - u0, E['log1-v'])
  if nDoc > 0:
    Elogp_pi_C = np.sum(E['logv']) + np.inner(kvec, E['log1-v'])
    Elogp_pi = np.inner(gamma * E['beta'] - 1, sumLogPi/nDoc)
    Elog_pmq_v =  Elog_pmq_v/nDoc
  else:
    Elogp_pi_C = 0
    Elogp_pi = 0

  elbo = Elogp_pi_C +  Elogp_pi + Elog_pmq_v
  return -1.0*elbo

def objGrad_u(u, sumLogPi, nDoc, gamma, alpha0):
  ''' Returns 2K-vector gradient of the constrained objective
      Args
      -------
      u := 2K-vector of real numbers, subject to 0 < u

      Returns
      -------
      g := 2K-vector of real numbers, 
            g[k] = gradient of -1 * L(u) with respect to u[k]
  '''
  assert not np.any(np.isnan(u))

  # UNPACK AND PREPARE building-block quantities
  u1, u0 = _unpack(u)
  K = u1.size
  E = _calcExpectations(u1, u0)
  dU1, dU0 = _calcGradients(u1, u0, E=E)

  kvec = K + 1 - np.arange(1, K+1)

  digammaU1 = digamma(u1)
  digammaU0 = digamma(u0)
  digammaBoth = digamma(u1+u0)

  dU1_Elogpmq_v = digammaU1 - digammaBoth \
                  + (1 - u1) * dU1['Elogv'] \
                  - E['logv'] \
                  + (alpha0 - u0) * dU1['Elog1-v']
  dU0_Elogpmq_v = digammaU0 - digammaBoth \
                  + (1 - u1) * dU0['Elogv'] \
                  - E['log1-v'] \
                  + (alpha0 - u0) * dU0['Elog1-v']

  if nDoc > 0:
    dU1_Elogp_pi = dU1['Elogv'] + kvec * dU1['Elog1-v'] \
                  + gamma * np.dot(dU1['Ebeta'], sumLogPi/nDoc)
    dU0_Elogp_pi = dU0['Elogv'] + kvec * dU0['Elog1-v'] \
                  + gamma * np.dot(dU0['Ebeta'], sumLogPi/nDoc)
    dU1_Elogpmq_v /= nDoc
    dU0_Elogpmq_v /= nDoc
  else:
    dU1_Elogp_pi = 0
    dU0_Elogp_pi = 0

  gvecU1 = dU1_Elogp_pi + dU1_Elogpmq_v
  gvecU0 = dU0_Elogp_pi + dU0_Elogpmq_v
  gvecU = np.hstack([gvecU1, gvecU0])
  return -1.0 * gvecU


########################################################### calcExpectations
########################################################### 
def _calcExpectations(u1, u0):
  ''' Calculate expectations of v and beta(v)
        under the model v[k] ~ Beta(U1[k], U0[k])
  '''
  E = dict()
  E['v'] = u1 / (u1 + u0)
  E['1-v'] = u0 / (u1 + u0)
  assert not np.any(np.isnan(E['v']))

  E['beta'] = v2beta(E['v'])

  digammaBoth = digamma(u1 + u0)
  E['logv'] = digamma(u1) - digammaBoth
  E['log1-v'] = digamma(u0) - digammaBoth
  return E

def _calcGradients(u1, u0, E):
  '''
  '''
  dU1 = dict()
  dU0 = dict()
  K = u1.size
  uboth = u1 + u0
  polygamma1Both = polygamma(1, uboth)
  dU1['Elogv'] = polygamma(1, u1) - polygamma1Both
  dU0['Elogv'] = -polygamma1Both
  dU1['Elog1-v'] = -polygamma1Both
  dU0['Elog1-v'] = polygamma(1,u0) - polygamma1Both

  Q1 = u1 / (uboth * uboth)
  Q0 = u0 / (uboth * uboth)

  dU1_Ebeta = np.tile(E['beta'], (K,1))
  dU1_Ebeta /= E['1-v'][:,np.newaxis]
  diagIDs = np.diag_indices(K)
  dU1_Ebeta[diagIDs] /= -E['v']/E['1-v']  

  # Slow way to force lower-triangle of dU1 to be all zeros
  #lowTriIDs = np.tril_indices(K, -1)
  #dU1_Ebeta[lowTriIDs] = 0

  # Fast way
  lowTriIDs = get_lowTriIDs(K)
  dU1_Ebeta[lowTriIDs] = 0

  # Fastest way
  #lowTriIDs = get_lowTriIDs_flat(K)
  #dU1_Ebeta.ravel()[flatlowTriIDs] = 0

  dU0_Ebeta = dU1_Ebeta * Q1[:,np.newaxis]
  dU1_Ebeta *= -1 * Q0[:,np.newaxis]

  dU1['Ebeta'] = dU1_Ebeta
  dU0['Ebeta'] = dU0_Ebeta
  return dU1, dU0


########################################################### Objective/gradient
########################################################### in terms of c

def objFunc_c(c, *args):
  ''' Returns scalar value of unconstrained objective function
      Args
      -------
      c := 2*K-vector of real numbers

      Returns
      -------
      f := -1 * L( v2c(c) ), where L is ELBO objective (log posterior)
  '''
  return objFunc_u(np.exp(c), *args)

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
  u = np.exp(c)
  dfdu = objGrad_u(u, *args)
  dudc = u
  dfdc = dfdu * dudc
  return dfdc


########################################################### Utility funcs
###########################################################
def _unpack(u):
  K = u.size/2
  u1 = u[:K]
  u0 = u[K:]
  return u1, u0

########################################################### Transform funcs
########################################################### u2v, u2beta
def u2v(u):
  u1, u0 = _unpack(u)
  return u1 / (u1 + u0)

def u2beta(u):
  u1, u0 = _unpack(u)
  v = u1 / (u1 + u0)
  return v2beta(v)

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
