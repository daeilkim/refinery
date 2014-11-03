'''
MergeMove.py

Merge components of a bnpy model.

Usage
--------
Inside a LearnAlg, to try a merge move on a particular model and dataset.
>>> hmodel, SS, curEv, MoveInfo = run_merge_move(hmodel, Data, SS)

To force a merge of components "2" and "4"
>>> hmodel, SS, curEv, MoveInfo = run_merge_move(hmodel, Data, SS, kA=2, kB=4)
'''

import numpy as np
from ..util import EPS, discrete_single_draw
import logging
from collections import defaultdict

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

def run_merge_move(curModel, Data, SS=None, curEv=None, doVizMerge=False,
                   kA=None, kB=None, excludeList=list(), 
                   excludePairs=defaultdict(lambda: set()),
                   mergename='marglik', randstate=np.random.RandomState(),
                   **kwargs):
  ''' Creates candidate model with two components merged,
      and returns either candidate or current model,
      whichever has higher log probability (ELBO).

      Args
      --------
       curModel : bnpy model whose components will be merged
       Data : bnpy Data object 
       SS : bnpy SuffStatDict object for Data under curModel
            must contain precomputed merge entropy in order to try a merge.
       curEv : current evidence bound, provided to save re-computation.
                curEv = curModel.calc_evidence(SS=SS)
       kA, kB : (optional) integer ids for which specific components to merge
       excludeList : (optional) list of integer ids excluded when selecting
                      which components to merge. useful when doing multiple 
                      rounds of merges, since precomputed merge terms are 
                      valid for one merge only.
      Returns
      --------
      hmodel, SS, evBound, MoveInfo

      hmodel := candidate or current model (bnpy HModel object)
      SS := suff stats for Data under hmodel
      evBound := log evidence (ELBO) of Data under hmodel
      MoveInfo := dict of info about this merge move, with fields
            didAccept := boolean flag, true if candidate accepted
            msg := human-readable string about this move
            kA, kB := indices of the components to be merged.
  ''' 
  if SS is None:
    LP = curModel.calc_local_params(Data)
    SS = curModel.get_global_suff_stats(Data, LP,
                                        doPrecompEntropy=True,
                                        doPrecompMerge=True)
  if curEv is None:
    curEv = curModel.calc_evidence(SS=SS)
    
  # Need at least two components to merge!
  if curModel.allocModel.K == 1:
    MoveInfo = dict(didAccept=0, msg="need >= 2 comps to merge")    
    return curModel, SS, curEv, MoveInfo  
  
  if not SS.hasMergeTerms():
    MoveInfo = dict(didAccept=0, msg="suff stats did not have merge terms")    
    return curModel, SS, curEv, MoveInfo  

  if kA in excludeList:
    MoveInfo = dict(didAccept=0, msg="target comp kA must be excluded.")    
    return curModel, SS, curEv, MoveInfo  
    
  # Select which 2 components kA, kB in {1, 2, ... K} to merge
  if kA is None or kB is None:
    kA, kB = select_merge_components(curModel, Data, SS,
                                     kA=kA, excludeList=excludeList, 
																		 excludePairs=excludePairs,
                                     mergename=mergename, 
                                     randstate=randstate)
  # Enforce that kA < kB. 
  # This step is essential for indexing mergeEntropy matrix, etc.
  assert kA != kB
  assert kA not in excludeList
  assert kB not in excludeList
  assert (kA,kB) not in excludePairs
  kMin = np.minimum(kA,kB)
  kB  = np.maximum(kA,kB)
  kA = kMin
  
  # Create candidate merged model
  propModel, propSS = propose_merge_candidate(curModel, SS, kA, kB)

  # Decide whether to accept the merge
  propEv = propModel.calc_evidence(SS=propSS)

  if np.isnan(propEv) or np.isinf(propEv):
    raise ValueError('propEv should never be nan/inf')
    
  if doVizMerge:
    viz_merge_proposal(curModel, propModel, kA, kB, curEv, propEv)

  if propEv > curEv:
    msg = "merge %3d & %3d | ev +%.3e ****" % (kA, kB, propEv - curEv)
    MoveInfo = dict(didAccept=1, kA=kA, kB=kB, msg=msg)
    return propModel, propSS, propEv, MoveInfo
  else:
    msg = "merge %3d & %3d | ev -%.3e" % (kA, kB, curEv - propEv)
    MoveInfo = dict(didAccept=0, kA=kA, kB=kB, msg=msg)
    return curModel, SS, curEv, MoveInfo


########################################################## Select kA,kB to merge
##########################################################
def select_merge_components(curModel, Data, SS, LP=None,
                            mergename='marglik', randstate=None,
                            kA=None, excludeList=[], 
														excludePairs=defaultdict(lambda:set())):
  ''' Select which two existing components to merge when constructing
      a candidate "merged" model from curModel, which has K components.
      We select components kA, kB by their integer ID, in {1, 2, ... K}

      Args
      --------
      curModel : bnpy model whose components we should merge
      Data : data object 
      SS : suff stats object for Data under curModel
      LP : local params dictionary (not required except for 'overlap')
      mergename : string specifying routine for how to select kA, kB
                  options include
                  'random' : select comps at random, without using data.
                  'marglik' : select comps by marginal likelihood ratio.
      Returns
      --------
      kA : integer id of the first component to merge
      kB : integer id of the 2nd component to merge

      This method guarantees that kA < kB.
  '''
  # Select routine for sampling component IDs kA, kB
  K = curModel.obsModel.K
  if mergename == 'random':
    ps = np.ones(K)
    ps[excludeList] = 0
    if kA is None:
      kA = discrete_single_draw(ps, randstate)
    ps[kA] = 0
    for kk in excludePairs[kA]:
      ps[kk] = 0
    kB = discrete_single_draw(ps, randstate)
  elif mergename == 'marglik':
    # Sample kA    
    # kA ~ Unif({1, 2, ... K})
    if kA is None:
      unifps = np.ones(K)
      unifps[excludeList] = 0
      kA = discrete_single_draw(unifps, randstate)
    # Sample kB
    # Pr(kb) \propto ratio of M(kA and kB) to M(kA)*M(kB)
    logmA = curModel.obsModel.calcLogMargLikForComp(SS, kA)  
    logscore = -1 * np.inf * np.ones(K)    
    for kB in xrange(K):
      if kB == kA or kB in excludeList or kB in excludePairs[kA]:
				continue
      logmB = curModel.obsModel.calcLogMargLikForComp(SS, kB)
      logmCombo = curModel.obsModel.calcLogMargLikForComp(SS, kA, kB)
      logscore[kB] = logmCombo - logmA - logmB
    if np.all(np.isinf(logscore)):
      ps = np.ones(K)
      ps[kA] = 0
      ps[excludeList] = 0
      ps[list(excludePairs[kA])] = 0
      if np.sum(ps) < EPS:
        raise ValueError("All possible choices excluded!")
    else:
      ps = np.exp(logscore - np.max(logscore))
    kB = discrete_single_draw(ps, randstate)
  else:
    raise NotImplementedError("Unknown mergename %s" % (mergename))
  # Here, we perform final validity checks on kA, kB
  # ensuring that kA < kB always
  kMin = np.minimum(kA,kB)
  kB  = np.maximum(kA,kB)
  kA = kMin
  assert kA < kB
  assert kB < K
  assert kA not in excludeList
  assert kB not in excludeList
  return kA, kB

############################################################ Construct new model
############################################################
def propose_merge_candidate(curModel, SS, kA=None, kB=None):
  ''' Propose new bnpy model from the provided current model (with K comps),
      where components kA, kB are combined into one "merged" component

      Returns
      --------
      propModel := bnpy HModel object
      propSS := bnpy sufficient statistic object.

      Both propSS and propModel have K-1 components.
  '''
  propModel = curModel.copy()
  
  # Rewrite candidate's kA component to be the merger of kA+kB
  # For now, **all* components get updated.
  # TODO: smartly avoid updating obsModel comps except related to kA/kB
  propSS = SS.copy()
  propSS.mergeComps(kA, kB)
  assert propSS.K == SS.K - 1
  propModel.update_global_params(propSS)

  # Remember, after calling update_global_params
  #  propModel's components must exactly match propSS's.
  # So kB has effectively been deleted here. It is already gone.
  return propModel, propSS


############################################################ Visualization
############################################################
def viz_merge_proposal(curModel, propModel, kA, kB, curEv, propEv):
  ''' Visualize merge proposal (in 2D)
  '''
  from ..viz import GaussViz, BarsViz
  from matplotlib import pylab
  
  fig = pylab.figure()
  h1 = pylab.subplot(1,2,1)
  if curModel.obsModel.__class__.__name__.count('Gauss'):
    GaussViz.plotGauss2DFromHModel(curModel, compsToHighlight=[kA, kB])
  else:
    BarsViz.plotBarsFromHModel(curModel, compsToHighlight=[kA, kB], figH=h1)
  pylab.title( 'Before Merge' )
  pylab.xlabel( 'ELBO=  %.2e' % (curEv) )
    
  h2 = pylab.subplot(1,2,2)
  if curModel.obsModel.__class__.__name__.count('Gauss'):
    GaussViz.plotGauss2DFromHModel(propModel, compsToHighlight=[kA])
  else:
    BarsViz.plotBarsFromHModel(propModel, compsToHighlight=[kA], figH=h2)
  pylab.title( 'After Merge' )
  pylab.xlabel( 'ELBO=  %.2e \n %d' % (propEv, propEv > curEv))
  pylab.show(block=False)
  try: 
    x = raw_input('Press any key to continue / Ctrl-C to quit >>')
  except KeyboardInterrupt:
    import sys
    sys.exit(-1)
  pylab.close()

