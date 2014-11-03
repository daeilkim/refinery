'''
MergeTracker.py

Object for tracking many back-to-back merge moves, making sure that
* no component that was already successfully merged is modified again
* no pair of components is tried more than once
'''
import numpy as np
from collections import defaultdict

class MergeTracker(object):

  def __init__(self, K):
    '''
    '''
    self.K = K
    self.OrigK = K
    self.nTrial = 0
    self.nSuccess = 0
    self.excludeList = set()
    self._initExcludePairs() # take care of self-loops
    self.acceptedIDs = list()
    self.acceptedOrigIDs = list()
    self.removedIDs = list()
    self.InfoLog = list()

  def addPairsToExclude(self, kAList, kBList):
    for ii in xrange(len(kAList)):
      kA = kAList[ii]
      kB = kBList[ii]
      self.excludePairs[kA].add(kB)
      self.excludePairs[kB].add(kA)
    self._synchronize_and_verify()

  def hasAvailablePairs(self):
    ''' Return True if not all candidates have been attempted, False otherwise
    '''
    # NOTE: this is way overkill
    if len(self.excludeList) > self.K - 2:
      return False
    candidates = self.getAvailableComps()
    if len(candidates) < 2:
      return False
    return True

  def hasAvailablePartnersForComp(self, kA):
    ''' Return True if some viable partners for kA remain, False otherwise
    '''
    # NOTE: this is way overkill
    if len(self.excludeList) > self.K - 2:
      return False
    candidates = self.getAvailablePartnersForComp(kA)
    if len(candidates) < 1:
      return False
    return True

  def verifyPair(self, kA, kB):
    ''' Verify that kA,kB make a valid merge pair that 
        * obeys kA < kB
        * has not already been attempted
        * has not already been excluded
             (by either one being party to a successful merge)
    '''
    assert 0 <= kA and kA < kB
    assert 0 <= kB and kB < self.K
    assert kA not in self.excludeList
    assert kB not in self.excludeList
    assert kA not in self.excludePairs[kB]
    assert kB not in self.excludePairs[kA]

  def getAvailablePartnersForComp(self, kA):
    ''' Get list of ids that can partner with component kA
    '''
    candidates = list()
    for x in range(self.K):
      if x not in self.excludeList:
        if x not in self.excludePairs[kA]:
          candidates.append(x)
    return candidates

  def getAvailableComps(self, kA=None):
    ''' Get list of ids for available components
    '''
    candidates = list()
    for x in range(self.K):
      if x not in self.excludeList:
        candidates.append(x)
    return candidates

  def recordResult(self, kA=None, kB=None, didAccept=False, msg=None, **kwargs):
    '''
    '''
    if kA is None or kB is None:
      return

    self.verifyPair(kA, kB)
    self.nTrial += 1
    self.InfoLog.append(msg)
    
    if didAccept:
      self.nSuccess += 1
      self._recordAcceptedMove(kA, kB)
    else:
      self.excludePairs[kA].add(kB)
      self.excludePairs[kB].add(kA)

    self._synchronize_and_verify()

  def _recordAcceptedMove(self, kA=0, kB=1):
    self._recordOriginalIDs(kA, kB)
    self.acceptedIDs.append((kA,kB))
    self.excludeList.add(kA)
    self._removeCompAndShiftDown(kB)
    

  def _recordOriginalIDs(self, kA, kB):
    for kk in reversed(self.removedIDs):
      if kA >= kk:
        kA += 1
      if kB >= kk:
        kB += 1
    self.acceptedOrigIDs.append((kA,kB))
        

  def _removeCompAndShiftDown(self, kB):
    ''' Call when comp kB has been "merged away",
          and we need to remove it and reindex the model 
          from K to K-1 components
    '''
    self.K -= 1
    self.removedIDs.append(kB)
    # ------------ shift down excludeList
    if kB in self.excludeList:
      raise ValueError('Bad! kB=%d should be excluded.' % (kB))
    self.excludeList = self._getReindexedSet(self.excludeList, kB)

    # ------------ shift down excludePairs
    newXPairs = defaultdict(lambda:set())
    for kk in self.excludePairs:
      kset = self.excludePairs[kk]
      if kB in kset:
        kset.remove(kB)
      kset = self._getReindexedSet(kset, kB)     
      if kk < kB:
        newXPairs[kk] = kset
      if kk > kB:
        newXPairs[kk-1] = kset
    self.excludePairs = newXPairs

  def _getReindexedSet(self, xset, kB):
    ''' Given set of integers A, return set B where
          if x in A, either x    in B (for x < kB)
                        or (x-1) in B (for x > kB)
    '''
    xarr = np.asarray(list(xset))
    xarr[xarr > kB] -= 1
    return set(list(xarr))

  def _initExcludePairs(self):
    '''
    '''
    self.excludePairs = defaultdict(lambda:set())
    for kk in range(self.K):
      self.excludePairs[kk].add(kk)

  def _synchronize_and_verify(self):
    ''' Make sure excludePairs and excludeList are up-to-date with each other
    '''
    for kx in list(self.excludeList):
      assert kx >= 0
      assert kx < self.K
      self.excludePairs[kx].update(range(self.K))
      for kk in xrange(self.K):
        self.excludePairs[kk].add(kx)

    for kk in range(self.K):
      kset = self.excludePairs[kk]
      # if kk has any more possible valid partners, 
      #  then set can have at most K-1 members (missing a partner)
      if len(kset) > self.K - 1: 
        self.excludeList.add(kk)

    for kk in self.excludePairs:
      assert kk >= 0
      assert kk < self.K
      karr = np.asarray(list(self.excludePairs[kk]))
      assert np.all(karr >= 0)
      assert np.all(karr < self.K)
