'''
FromScratchMult.py

Initialize params of HModel with multinomial observations from scratch.
'''
import numpy as np
from scipy.special import digamma
from scipy.cluster import vq

hasRexAvailable = True
try:
  import KMeansRex
except ImportError:
  hasRexAvailable = False

def init_global_params(hmodel, Data, initname='randexamples',
                               seed=0, K=0, initarg=None, **kwargs):
  ''' Initialize hmodel's global parameters in-place.

      Returns
      -------
      Nothing. hmodel is updated in place.
  '''    

  doc_range = Data.doc_range
  PRNG = np.random.RandomState(seed)
  LP = None
  if initname == 'randexamples':
    ''' Choose K documents at random
    '''
    DocWord = Data.to_sparse_docword_matrix()
    chosenDocIDs = PRNG.choice(Data.nDoc, K, replace=False)
    PhiTopicWord = np.asarray(DocWord[chosenDocIDs].todense())
    PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    beta = np.ones(K)
    hmodel.set_global_params(K=K, beta=beta, topics=PhiTopicWord)
    return
  elif initname == 'randomfromarg':
    ''' Draw expected topic-word probability vectors 
          from a Dirichlet with symmetric parameter initarg
    '''
    PhiTopicWord = PRNG.gamma( initarg, 1., (K,Data.vocab_size))
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    beta = np.ones(K)
    hmodel.set_global_params(K=K, beta=beta, topics=PhiTopicWord)
    return
  elif initname == 'randomfromprior':
    ''' Draw expected topic-word probability vectors from their prior
    '''
    lvec = hmodel.obsModel.obsPrior.lamvec
    PhiTopicWord = PRNG.gamma(lvec, 1., (K,Data.vocab_size))
    PhiTopicWord += 1e-200
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    beta = np.ones(K)
    hmodel.set_global_params(K=K, beta=beta, topics=PhiTopicWord)
    return
  elif initname == 'randsoftpartition':
    ''' Assign responsibility for K topics at random to all words
    '''        
    word_variational = PRNG.rand(Data.nObs, K)
    word_variational /= word_variational.sum(axis=1)[:,np.newaxis]
    doc_variational = np.zeros((Data.nDoc, K))
    for d in xrange(Data.nDoc):
      start,stop = doc_range[d,:]
      doc_variational[d,:] = np.sum(word_variational[start:stop,:],axis=0)
    # no return, instead uses final M-step below this if block
  elif initname == 'plusplus':
    if not hasRexAvailable:
      raise NotImplementedError("KMeansRex must be on python path")
    DocWord = Data.to_sparse_docword_matrix().toarray()
    PhiTopicWord = KMeansRex.SampleRowsPlusPlus(DocWord, K, seed=seed)
    PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    beta = np.ones(K)
    hmodel.set_global_params(K=K, beta=beta, topics=PhiTopicWord)
    return
  elif initname == 'kmeansplusplus':
    if not hasRexAvailable:
      raise NotImplementedError("KMeansRex must be on python path")
    DocWord = Data.to_sparse_docword_matrix().toarray()
    PhiTopicWord, Z = KMeansRex.RunKMeans(DocWord, K, initname='plusplus',
                                          Niter=10, seed=seed)
    PhiTopicWord += 0.01 * PRNG.rand(K, Data.vocab_size)
    PhiTopicWord /= PhiTopicWord.sum(axis=1)[:,np.newaxis]
    beta = np.ones(K)
    hmodel.set_global_params(K=K, beta=beta, topics=PhiTopicWord)
    return
  else:
    raise NotImplementedError('Unrecognized initname ' + initname)
    
  # Final step : (only some initname routines make it here)
  # Take provided local params LP,
  #     and perform M-step on the provided Data
  # This sets hmodel's global parameters to valid values consistent with Data
  if LP is None:
    if hmodel.getAllocModelName().count('HDP') > 0:
      LP = getLPfromResp(word_variational, Data)
    else:
      # AdmixModel case
      LP = dict(doc_variational=doc_variational, 
                word_variational=word_variational)

  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)

def getLPfromResp(Resp, Data, smoothMass=0.01):
  ''' Returns local parameters (LP) for an HDP model
        given word-level responsibility matrix and Data (which indicates num docs)
      Returns
      --------
      LP : dict with fields 
              word_variational
              alphaPi
              DocTopicCount
              E_logPi
  '''
  D = Data.nDoc
  K = Resp.shape[1]
  DocTopicC = np.zeros((D, K))
  for dd in range(D):
    start,stop = Data.doc_range[dd,:]
    DocTopicC[dd,:] = np.dot(Data.word_count[start:stop],        
                               Resp[start:stop,:]
                             )
  # Alpha and ElogPi : D x K+1 matrices
  padCol = smoothMass * np.ones((D,1))
  alph = np.hstack( [DocTopicC + smoothMass, padCol])    
  ElogPi = digamma(alph) - digamma(alph.sum(axis=1))[:,np.newaxis]
  assert ElogPi.shape == (D,K+1)
  return dict(word_variational =Resp, 
              E_logPi=ElogPi, alphaPi=alph,
              DocTopicCount=DocTopicC)
