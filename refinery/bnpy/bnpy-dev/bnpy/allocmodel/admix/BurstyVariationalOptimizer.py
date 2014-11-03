'''
BurstyVariationalOptimizer.py

Functions for variational approximation to Global Topics in Observation Model

Model Notation 
--------
Compound Dirichlet with fixed truncation K
phi_k ~ Dir(tau)

for each document d, draw eta (document-specific topic x word dist.)
eta_dk ~ Dir(nu * phi_k)

W is the number of unique vocabulary words
K is the number of Topics

For input we will need:
tau = concentration parameter at the global topic level
nu = concentration parameter at the document level
lambda_k = W x 1 vector of variational posteriors
Elog_eta_k = W x 1 vector of sum across all documents for expected log suff. stats 
                  for topic k at the document level. Should be precomputed once before calling this

Notes
-------
Relies on approximation to E[ log norm const of Dirichlet],
  which requires parameter nu < 1

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

EPS = 10*np.finfo(float).eps

def estimate_lambda_k(nu=0.1, nDoc=0, K=2, lambda_k=None, tau=None, method='l_bfgs', **kwargs):
  ''' Solve optimization problem to estimate parameters u
      for the approximate posterior on stick-breaking fractions v
      q(Omega_kw | lambda_kw ) = Dir( Omega_kw | lambda_kw)

      Returns
      -------
      lambda_k, a 1 x W vector representing global topic k x word frequencies
  '''
  W = lambda_k.size 
  init_lambda_k = 0.01 * np.ones(W)
  
  myFunc = lambda lambda_k: objectiveFunc(lambda_k, nu, Elog_eta_k, nDoc)
  myGrad = lambda lambda_k: objectiveGradient(lambda_k, nu, tau, Elog_eta_k, nDoc)
  best_lambda_k_unconstrained, bestf, Info = scipy.optimize.fmin_l_bfgs_b(myFunc, np.log(init_lambda_k), fprime=myGrad, disp=None)
  best_lambda_k = np.exp(best_lambda_k_log_unconstrained)
  # Taking the exponenent to reverse our use of the log
  return best_lambda_k

def objectiveFunc(lambda_k, nu, tau, Elog_eta_k, nDoc):
  ''' Calculate unconstrained objective function for HDP variational learning
  '''
  assert not np.any(np.isnan(Cvec))
  assert not np.any(np.isinf(Cvec))
  W = lambda_k.size

  # PREPARE building-block expectations
  digammaAll = digamma(np.sum(lambda_k))
  Elog_phi_k = digamma(lambda_k) - digammaAll
  E_phi_k = lambda_k / np.sum(lambda_k)

  # CALCULATE each term in the function
  prior = (tau-1) * np.sum(Elog_phi_k)
  likelihood = nDoc * np.sum(Elog_phi_k) \
             + nu * np.sum( E_phi_k * Elog_eta_k )

  entropy = gammaln( np.sum(lambda_k) ) \
                - np.sum( gammaln(lambda_k) ) \
                + np.sum((lambda_k - 1) * Elog_phi_k)

  f = -1 * (prior + likelihood - entropy)
  return f

def objectiveGradient(lambda_k, nu, tau, Elog_eta_k, nDoc):
  ''' Calculate gradient of objectiveFunc, objective for HDP variational 
      Returns
      -------
        gvec : 2*K length vector,
              where each entry gives partial derivative with respect to
                  the corresponding entry of Cvec
  '''
  # lvec is the derivative of log(lambda_k) via chain rule
  lvec = 1/(lambda_k)
  W = lvec.size
  
  # Derivative of log eta
  digammaAll = digamma(np.sum(lambda_k))
  Elog_lambda_k = digamma(lambda_k) - digammaAll

  # Derivative of Elog_phi_k and E_phi_k
  polygammaAll = polygamma(1,np.sum(lambda_k))
  dElog_phi_k = polygamma(1,lambda_k) - polygammaAll
  lambda_k_sum = np.sum(lambda_k)
  dE_phi_k = (lambda_k_sum - lambda_k) / np.power(lambda_k_sum,2)

  gvec = dElog_phi_k * (N + tau - lambda_k) \
       + dE_phi_k * nu * Elog_eta_k
  gvec = -1 * gvec

  # Apply chain rule!
  gvecC = lvec * gvec
  return gvecC

