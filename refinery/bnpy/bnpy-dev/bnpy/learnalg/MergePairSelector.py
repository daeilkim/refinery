'''
MergePairSelector

Wrapper class for routines that randomly select which components kA, kB 
  to attempt to merge

USAGE
>> MTracker = MergeTracker(SS.K)
>> MSelector = MergePairSelector(hmodel, SS)
>> kA, kB = MSelector.select_merge_component(MTracker)
'''

import numpy as np
from ..util import choice

class MergePairSelector(object):

  def __init__(self):
    self.MScores = dict()
    self.PairMScores = dict()

  def select_merge_components(self, hmodel, SS, MTracker, kA=None, 
                                mergename='marglik', randstate=np.random):
    if mergename == 'random':
      kA, kB = self._drawPair_random(MTracker, kA=kA, randstate=randstate)
    elif mergename == 'marglik':
      kA, kB = self._drawPair_marglik(hmodel, SS, 
                                      MTracker, kA=kA, randstate=randstate)
    else:
      raise NotImplementedError("Unknown mergename %s" % (mergename))
    # Ensure always that kA < kB always
    kMin = np.minimum(kA,kB)
    kB  = np.maximum(kA,kB)
    kA = kMin
    MTracker.verifyPair(kA, kB)
    return kA, kB
  
  def reindexAfterMerge(self, kA, kB):
    newMScores = dict()
    for kk in self.MScores:
      if kk > kB:
        newMScores[kk-1] = self.MScores[kk]
      elif kk < kB:
        newMScores[kk] = self.MScores[kk]
    self.MScores = newMScores

    newPScores = dict()
    for k1, k2 in self.PairMScores:
      if k1 == kB or k2 == kB:
        continue
      newk1 = k1
      newk2 = k2
      if k1 > kB:
        newk1 -= 1
      if k2 > kB:
        newk2 -= 1
      newPScores[ (newk1,newk2)] = self.PairMScores[(k1,k2)]
    self.PairMScores = newPScores

  def _drawPair_random(self, MTracker, kA=None, randstate=np.random):
    '''
        Returns
        --------
        kA
        kB
    '''
    # --------- Select kA
    candidatesA = MTracker.getAvailableComps()
    nA = len(candidatesA)
    pA = np.ones(nA)/nA
    if kA is None:
      kA = choice(candidatesA, ps=pA, randstate=randstate)
      #kA = randstate.choice(candidatesA, p=pA) # uniform draw
    else:
      assert kA in candidatesA
    # --------- Select kB | kA
    candidatesB = MTracker.getAvailablePartnersForComp(kA)
    assert len(candidatesB) > 0
    nB = len(candidatesB)
    pB = np.ones(nB)/nB
    kB = choice(candidatesB, ps=pB, randstate=randstate)
    #kB = randstate.choice(candidatesB, p=pB) # uniform draw
    return kA, kB

  def _drawPair_marglik(self, hmodel, SS,
                              MTracker, kA=None, randstate=np.random):
    '''
        Returns
        --------
        kA
        kB
    '''
    # --------- Select kA
    candidatesA = MTracker.getAvailableComps()
    if kA is None:
      nA = len(candidatesA)
      pA = np.ones(nA)/nA
      kA = choice(candidatesA, ps=pA, randstate=randstate)
      #kA = randstate.choice(candidatesA, p=pA) # uniform draw
    else:
      assert kA in candidatesA
    # --------- Select kB | kA
    candidatesB = MTracker.getAvailablePartnersForComp(kA)
    assert len(candidatesB) > 0
    ps = self._calcMargLikProbVector(hmodel, SS, kA, candidatesB)
    kB = choice(candidatesB, ps=ps, randstate=randstate)
    #kB = randstate.choice(candidatesB, p=ps)
    return kA, kB

  def _calcMargLikProbVector(self, hmodel, SS, kA, candidates):
    logps = list()
    for kB in candidates:
      logps.append( self._calcMScoreForCandidatePair(hmodel, SS, kA, kB) )
    logps = np.asarray(logps)
    ps = np.exp(logps - np.max(logps))
    ps /= np.sum(ps)
    return ps

  def _calcMScoreForCandidatePair(self, hmodel, SS, kA, kB):
    logmA = self._calcLogMargLikForComp(hmodel, SS, kA)
    logmB = self._calcLogMargLikForComp(hmodel, SS, kB)
    logmAB = self._calcLogMargLikForPair(hmodel, SS, kA, kB)
    return logmAB - logmA - logmB

  def _calcLogMargLikForComp(self, hmodel, SS, kA):
    if kA in self.MScores:
      return self.MScores[kA]
    mA = hmodel.obsModel.calcLogMargLikForComp(SS, kA, doNormConstOnly=True)  
    self.MScores[kA] = mA
    return mA

  def _calcLogMargLikForPair(self, hmodel, SS, kA, kB):
    if (kA,kB) in self.PairMScores:
      return self.PairMScores[ (kA,kB)]
    elif (kB,kA) in self.PairMScores:
      return self.PairMScores[ (kB,kA)]
    else:
      mAB = hmodel.obsModel.calcLogMargLikForComp(SS, kA, kB, doNormConstOnly=True)  
      self.PairMScores[(kA,kB)] = mAB
      return mAB