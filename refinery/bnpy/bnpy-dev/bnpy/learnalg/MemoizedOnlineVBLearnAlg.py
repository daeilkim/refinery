'''
MemoizedOnlineVBLearnAlg.py

Implementation of Memoized Online VB (moVB) learn alg for bnpy models
'''
import numpy as np
import joblib
import os
import logging
from collections import defaultdict

from bnpy.util import isEvenlyDivisibleFloat
import BirthMove
import MergeMove
from LearnAlg import LearnAlg
from bnpy.suffstats import SuffStatBag

Log = logging.getLogger('bnpy')


class MemoizedOnlineVBLearnAlg(LearnAlg):

  def __init__( self, **kwargs):
    ''' Creates memoized VB learning algorithm object,
          including specialized internal fields to hold "memoized" statistics
    '''
    super(type(self),self).__init__(**kwargs)
    self.SSmemory = dict()
    self.LPmemory = dict()
    if self.hasMove('merge'):
      self.MergeLog = list()
    if self.hasMove('birth'):
      # Track subsampled data aggregated across batches
      #self.targetDataList = list()
      # Track the components freshly added in current lap
      self.BirthCompIDs = list()
      self.ModifiedCompIDs = list()
      # Track the number of laps since birth last attempted
      #  at each component, to encourage trying diversity
      self.LapsSinceLastBirth = defaultdict(int)

  def fit(self, hmodel, DataIterator):
    ''' Run moVB learning algorithm, fit parameters of hmodel to Data,
          traversed one batch at a time from DataIterator

        Returns
        --------
        LP : None type, cannot fit all local params in memory
        Info : dict of run information, with fields
              evBound : final ELBO evidence bound
              status : str message indicating reason for termination
                        {'converged', 'max passes exceeded'}
    
    '''
    # Define how much of data we see at each mini-batch
    nBatch = float(DataIterator.nBatch)
    self.lapFracInc = 1.0/nBatch
    # Set-up progress-tracking variables
    iterid = -1
    lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0/nBatch)
    if lapFrac > 0:
      # When restarting an existing run,
      #  need to start with last update for final batch from previous lap
      DataIterator.lapID = int(np.ceil(lapFrac)) - 1
      DataIterator.curLapPos = nBatch - 2
      iterid = int(nBatch * lapFrac) - 1

    # memoLPkeys : keep list of params that should be retained across laps
    self.memoLPkeys = hmodel.allocModel.get_keys_for_memoized_local_params()
    mPairIDs = None

    BirthPlans = list()
    BirthResults = None
    prevBirthResults = None

    SS = None
    isConverged = False
    prevBound = -np.inf
    self.set_start_time_now()
    while DataIterator.has_next_batch():

      # Grab new data
      Dchunk = DataIterator.get_next_batch()
      batchID = DataIterator.batchID
      
      # Update progress-tracking variables
      iterid += 1
      lapFrac = (iterid + 1) * self.lapFracInc
      self.set_random_seed_at_lap(lapFrac)

      # M step
      if self.algParams['doFullPassBeforeMstep']:
        if SS is not None and lapFrac > 1.0:
          hmodel.update_global_params(SS)
      else:
        if SS is not None:
          hmodel.update_global_params(SS)
      
      # Birth move : track birth info from previous lap
      if self.isFirstBatch(lapFrac):
        if self.hasMove('birth') and self.do_birth_at_lap(lapFrac - 1.0):
          prevBirthResults = BirthResults
        else:
          prevBirthResults = list()

      # Birth move : create new components
      if self.hasMove('birth') and self.do_birth_at_lap(lapFrac):
        if self.doBirthWithPlannedData(lapFrac):
          hmodel, SS, BirthResults = self.birth_create_new_comps(
                                            hmodel, SS, BirthPlans)

        if self.doBirthWithDataFromCurrentBatch(lapFrac):
          hmodel, SS, BirthRes = self.birth_create_new_comps(
                                            hmodel, SS, Data=Dchunk)
          BirthResults.extend(BirthRes)

        self.BirthCompIDs = self.birth_get_all_new_comps(BirthResults)
        self.ModifiedCompIDs = self.birth_get_all_modified_comps(BirthResults)
      else:
        BirthResults = list()
        self.BirthCompIDs = list() # no births = no new components
        self.ModifiedCompIDs = list()

      # Select which components to merge
      if self.hasMove('merge') and not self.algParams['merge']['doAllPairs']:
        if self.isFirstBatch(lapFrac):
          if self.hasMove('birth'):
            compIDs = self.BirthCompIDs
          else:
            compIDs = []
          mPairIDs = MergeMove.preselect_all_merge_candidates(hmodel, SS, 
                           randstate=self.PRNG, compIDs=compIDs,
                           **self.algParams['merge'])

      # E step
      if batchID in self.LPmemory:
        oldLPchunk = self.load_batch_local_params_from_memory(
                                           batchID, prevBirthResults)
        LPchunk = hmodel.calc_local_params(Dchunk, oldLPchunk,
                                           **self.algParamsLP)
      else:
        LPchunk = hmodel.calc_local_params(Dchunk, **self.algParamsLP)

      # Collect target data for birth
      if self.hasMove('birth') and self.do_birth_at_lap(lapFrac+1.0):
        if self.isFirstBatch(lapFrac):
          BirthPlans = self.birth_select_targets_for_next_lap(
                                hmodel, SS, BirthResults)
        BirthPlans = self.birth_collect_target_subsample(
                                Dchunk, LPchunk, BirthPlans)
      else:
        BirthPlans = list()

      # Suff Stat step
      if batchID in self.SSmemory:
        SSchunk = self.load_batch_suff_stat_from_memory(batchID, SS.K)
        SS -= SSchunk

      SSchunk = hmodel.get_global_suff_stats(Dchunk, LPchunk,
                       doPrecompEntropy=True, 
                       doPrecompMergeEntropy=self.hasMove('merge'),
                       mPairIDs=mPairIDs,
                       )

      if SS is None:
        SS = SSchunk.copy()
      else:
        assert SSchunk.K == SS.K
        SS += SSchunk

      # Store batch-specific stats to memory
      if self.algParams['doMemoizeLocalParams']:
        self.save_batch_local_params_to_memory(batchID, LPchunk)          
      self.save_batch_suff_stat_to_memory(batchID, SSchunk)  

      # Handle removing "extra mass" of fresh components
      #  to make SS have size exactly consistent with entire dataset
      if self.hasMove('birth') and self.isLastBatch(lapFrac):
        hmodel, SS = self.birth_remove_extra_mass(hmodel, SS, BirthResults)

      # ELBO calc
      #self.verify_suff_stats(Dchunk, SS, lapFrac)
      evBound = hmodel.calc_evidence(SS=SS)

      # Merge move!      
      if self.hasMove('merge') and isEvenlyDivisibleFloat(lapFrac, 1.):
        hmodel, SS, evBound = self.run_merge_move(hmodel, SS, evBound, mPairIDs)

      # Save and display progress
      self.add_nObs(Dchunk.nObs)
      self.save_state(hmodel, iterid, lapFrac, evBound)
      self.print_state(hmodel, iterid, lapFrac, evBound)
      self.eval_custom_func(hmodel, iterid, lapFrac)

      # Check for Convergence!
      #  evBound will increase monotonically AFTER first lap of the data 
      #  verify_evidence will warn if bound isn't increasing monotonically
      if lapFrac > self.algParams['startLap'] + 1.0:
        isConverged = self.verify_evidence(evBound, prevBound, lapFrac)
        if isConverged and lapFrac > 5 and not self.hasMove('birth'):
          break
      prevBound = evBound

    # Finally, save, print and exit
    if isConverged:
      msg = "converged."
    else:
      msg = "max passes thru data exceeded."
    self.save_state(hmodel, iterid, lapFrac, evBound, doFinal=True) 
    self.print_state(hmodel, iterid, lapFrac,evBound,doFinal=True,status=msg)
    return None, self.buildRunInfo(evBound, msg)

  def verify_suff_stats(self, Dchunk, SS, lap):
    ''' Run-time checks to make sure the suff stats
        have expected values
    '''
    if self.savedir is not None:
      SSfile = os.path.join(self.savedir, 'SSdump-Lap%03d.dat' % (lap))
      if self.isLastBatch(lap):
        joblib.dump(SS, SSfile)
    if hasattr(Dchunk, 'nDocTotal') and Dchunk.nDocTotal < 4000:
      if self.hasMove('birth') and self.do_birth_at_lap(lap):
        if self.algParams['birth']['earlyLap'] > 0:
          pass
        elif lap < np.ceil(lap):
          assert SS.nDoc - Dchunk.nDocTotal > -0.001
        else:
          if abs(SS.nDoc - Dchunk.nDocTotal) > 0.01:
            print "WARNING @ lap %.2f | SS.nDoc=%d, nDocTotal=%d" % (lap, SS.nDoc, Dchunk.nDocTotal)
          assert abs(SS.nDoc - Dchunk.nDocTotal) < 0.01
      elif lap >= 1.0:
        assert abs(SS.nDoc - Dchunk.nDocTotal) < 0.01

    if hasattr(SS, 'N'):
      if not np.all(SS.N >= -1e-9):
        raise ValueError('N should be >= 0!')
      SS.N[SS.N < 0] = 0

  ######################################################### Load from memory
  #########################################################
  def load_batch_suff_stat_from_memory(self, batchID, K):
    ''' Load the suff stats stored in memory for provided batchID
        Returns
        -------
        SSchunk : bnpy SuffStatDict object for batchID
    '''
    SSchunk = self.SSmemory[batchID]
    # "replay" accepted merges from end of previous lap 
    if self.hasMove('merge'): 
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        if kA < SSchunk.K and kB < SSchunk.K:
          SSchunk.mergeComps(kA, kB)
      if SSchunk.hasMergeTerms():
        SSchunk.setMergeFieldsToZero()
    # "replay" generic and batch-specific births from this lap
    if self.hasMove('birth'):   
      Kextra = K - SSchunk.K
      if Kextra > 0:
        SSchunk.insertEmptyComps(Kextra)
    assert SSchunk.K == K
    return SSchunk  

  def load_batch_local_params_from_memory(self, batchID, BirthResults):
    ''' Load local parameter dict stored in memory for provided batchID
        Ensures "fast-forward" so that all recent merges/births
          are accounted for in the returned LP
        Returns
        -------
        LPchunk : bnpy local parameters dictionary for batchID
    '''
    LPchunk = self.LPmemory[batchID]
    if self.hasMove('birth') and LPchunk is not None:
      if BirthResults is not None and len(BirthResults) > 0:
        # new components have been "born", so discard old results
        #   since they no longer matter
        LPchunk = None

    if self.hasMove('merge') and LPchunk is not None:
      for MInfo in self.MergeLog:
        kA = MInfo['kA']
        kB = MInfo['kB']
        for key in self.memoLPkeys:
          if kB >= LPchunk[key].shape[1]:
            # Birth occured in previous lap, after this batch was visited.
            return None
          LPchunk[key][:,kA] = LPchunk[key][:,kA] + LPchunk[key][:,kB]
          LPchunk[key] = np.delete(LPchunk[key], kB, axis=1)
    return LPchunk

  ######################################################### Save to memory
  #########################################################
  def save_batch_suff_stat_to_memory(self, batchID, SSchunk):
    ''' Store the provided suff stats into the "memory" for later retrieval
    '''
    self.SSmemory[batchID] = SSchunk

  def save_batch_local_params_to_memory(self, batchID, LPchunk):
    ''' Store certain fields of the provided local parameters dict
          into "memory" for later retrieval.
        Fields to save determined by the memoLPkeys attribute of this alg.
    '''
    allkeys = LPchunk.keys()
    for key in allkeys:
      if key not in self.memoLPkeys:
        del LPchunk[key]
    if len(LPchunk.keys()) > 0:
      self.LPmemory[batchID] = LPchunk
    else:
      self.LPmemory[batchID] = None


  ######################################################### Birth moves!
  #########################################################
  def doBirthWithPlannedData(self, lapFrac):
    return self.isFirstBatch(lapFrac)

  def doBirthWithDataFromCurrentBatch(self, lapFrac):
    if self.isLastBatch(lapFrac):
      return False
    rem = lapFrac - np.floor(lapFrac)
    isWithinFrac = rem <= self.algParams['birth']['batchBirthFrac'] + 1e-6
    isWithinLimit = lapFrac <= self.algParams['birth']['batchBirthLapLimit'] 
    return isWithinFrac and isWithinLimit

  def birth_create_new_comps(self, hmodel, SS, BirthPlans=list(), Data=None):
    ''' Create new components 

        Returns
        -------
        hmodel : bnpy HModel, with (possibly) new components
        SS : bnpy SuffStatBag, with (possibly) new components
        BirthResults : list of dictionaries, one entry per birth move
                        each entry has fields
                        * TODO
    '''
    if Data is not None:
      if hasattr(Data, 'nDoc'):
        wordPerDocThr = self.algParams['birth']['birthWordsPerDocThr']
        if wordPerDocThr > 0:
          nWordPerDoc = np.asarray(Data.to_sparse_docword_matrix().sum(axis=1))
          candidates = nWordPerDoc >= wordPerDocThr
          candidates = np.flatnonzero(candidates)
        else:
          candidates = None
        targetData = Data.get_random_sample(
                                self.algParams['birth']['maxTargetSize'],
                                randstate=self.PRNG, candidates=candidates)
      else:
        targetData = Data.get_random_sample(
                                self.algParams['birth']['maxTargetObs'],
                                randstate=self.PRNG)


      Plan = dict(Data=targetData, ktarget=-1)
      BirthPlans = [Plan]

    nMoves = len(BirthPlans)
    BirthResults = list()
    for moveID, Plan in enumerate(BirthPlans):
      # Unpack data for current move
      ktarget = Plan['ktarget']
      targetData = Plan['Data']

      if ktarget is None or targetData is None:
        msg = Plan['msg']

      elif targetData.nObs < self.algParams['birth']['minTargetObs']:
        # Verify targetData large enough that birth would be productive
        msg = "BIRTH skipped. Target data too small (size %d)"
        msg = msg % (targetData.nObs)
      elif hasattr(targetData, 'nDoc') \
           and targetData.nDoc < self.algParams['birth']['minTargetSize']:
        msg = "BIRTH skipped. Target data too small (size %d)"
        msg = msg % (targetData.nDoc)

      else:
        hmodel, SS, MoveInfo = BirthMove.run_birth_move(
                 hmodel, targetData, SS, randstate=self.PRNG, 
                 ktarget=ktarget, **self.algParams['birth'])
        msg = MoveInfo['msg']
        if MoveInfo['didAddNew']:
          BirthResults.append(MoveInfo)

          for kk in MoveInfo['birthCompIDs']:
            self.LapsSinceLastBirth[kk] = -1

      if Data is None:
          self.print_msg( "%d/%d %s" % (moveID+1, nMoves, msg) )
      else:
          self.print_msg( "%d/%d BATCH %s" % (moveID+1, nMoves, msg) )

    return hmodel, SS, BirthResults

  def birth_remove_extra_mass(self, hmodel, SS, BirthResults):
    ''' Adjust hmodel and suff stats to remove the "extra mass"
          added during a birth move to brand-new components.
        After this call, SS should have scale exactly consistent with 
          the entire dataset (all B batches).

        Returns
        -------
        hmodel : bnpy HModel
        SS : bnpy SuffStatBag
    '''
    didChangeSS = False
    for MoveInfo in BirthResults:
      if MoveInfo['didAddNew']:
        extraSS = MoveInfo['extraSS']
        compIDs = MoveInfo['modifiedCompIDs']
        assert extraSS.K == len(compIDs)
        SS.subtractSpecificComps(extraSS, compIDs)
        didChangeSS = True
    if didChangeSS:
      hmodel.update_global_params(SS)
    return hmodel, SS

  def birth_select_targets_for_next_lap(self, hmodel, SS, BirthResults):
    ''' Create plans for next lap's birth moves
    
        Returns
        -------
        BirthPlans : list of dicts, 
                     each entry represents the plan for one future birth move
    '''
    if SS is not None:
      assert hmodel.allocModel.K == SS.K
    K =  hmodel.allocModel.K

    # Update counter for which components haven't been updated in a while
    for kk in range(K):
      self.LapsSinceLastBirth[kk] += 1

    # Ignore components that have just been added to the model.
    excludeList = self.birth_get_all_new_comps(BirthResults)

    # For each birth move, create a "plan"
    BirthPlans = list()
    for posID in range(self.algParams['birth']['birthPerLap']):
      try:
        ktarget = BirthMove.select_birth_component(SS, K=K, 
                          randstate=self.PRNG,
                          excludeList=excludeList, doVerbose=False,
                          lapsSinceLastBirth=self.LapsSinceLastBirth,
                          **self.algParams['birth'])
        self.LapsSinceLastBirth[ktarget] = 0
        excludeList.append(ktarget)
        Plan = dict(ktarget=ktarget, Data=None)
      except BirthMove.BirthProposalError, e:
        Plan = dict(ktarget=None, Data=None, msg=str(e))

      BirthPlans.append(Plan)
    return BirthPlans

  def birth_collect_target_subsample(self, Dchunk, LPchunk, BirthPlans):
    ''' Collect subsample of the data in Dchunk, and add that subsample
          to overall targeted subsample stored in input list BirthPlans
        This overall sample is aggregated across many batches of data.
        Data from Dchunk is only collected if more data is needed.

        Returns
        -------
        BirthPlans : list of planned births for the next lap,
                      updated to include data from Dchunk if needed
    '''
    import BirthMove
    
    for Plan in BirthPlans:
      # Skip this move if component selection failed
      if Plan['ktarget'] is None:
        continue

      birthParams = dict(**self.algParams['birth'])
      # Skip collection if have enough data already
      if Plan['Data'] is not None:
        if hasattr(Plan['Data'], 'nDoc'):
          if Plan['Data'].nDoc >= self.algParams['birth']['maxTargetSize']:
            continue
          birthParams['maxTargetSize'] -= Plan['Data'].nDoc
        else:
          if Plan['Data'].nObs >= self.algParams['birth']['maxTargetObs']:
            continue

      # Sample data from current batch, if more is needed
      targetData = BirthMove.subsample_data(Dchunk, LPchunk,
                          Plan['ktarget'], randstate=self.PRNG,
                          **birthParams)
      # Update Data for current entry in self.targetDataList
      if targetData is None:
        if Plan['Data'] is None:
          Plan['msg'] = "TargetData: No samples for target comp found."
      else:
        if Plan['Data'] is None:
          Plan['Data'] = targetData
        else:
          Plan['Data'].add_data(targetData)
        Plan['msg'] = "TargetData: nObs %d" % (Plan['Data'].nObs)

    return BirthPlans

  def birth_get_all_new_comps(self, BirthResults):
    ''' Returns list of integer ids of all new components added by
          birth moves summarized in BirthResults

        Returns
        -------
        birthCompIDs : list of integers, each entry is index of a new component
    '''
    birthCompIDs = list()
    for MoveInfo in BirthResults:
      birthCompIDs.extend(MoveInfo['birthCompIDs'])
    return birthCompIDs

  def birth_get_all_modified_comps(self, BirthResults):
    ''' Returns list of integer ids of all new components added by
          birth moves summarized in BirthResults

        Returns
        -------
        mCompIDs : list of integers, each entry is index of modified comp
    '''
    mCompIDs = list()
    for MoveInfo in BirthResults:
      mCompIDs.extend(MoveInfo['modifiedCompIDs'])
    return mCompIDs

  def birth_count_new_comps(self, BirthResults):
    ''' Returns total number of new components added by moves 
          summarized by BirthResults
    
        Returns
        -------
        Kextra : int number of components added by given list of moves
    '''
    Kextra = 0
    for MoveInfo in BirthResults:
      Kextra += len(MoveInfo['birthCompIDs'])
    return Kextra


  ######################################################### Merge moves!
  #########################################################
  def run_merge_move(self, hmodel, SS, evBound, mPairIDs=None):
    if self.algParams['merge']['version'] > 0:
      return self.run_merge_move_NEW(hmodel, SS, evBound, mPairIDs)
    else:
      return self.run_merge_move_OLD(hmodel, None, SS, evBound)

  def run_merge_move_NEW(self, hmodel, SS, evBound, mPairIDs=None):
    ''' Run (potentially many) merge moves on hmodel,
          performing necessary bookkeeping to
            (1) avoid trying the same merge twice
            (2) avoid merging a component that has already been merged,
                since the precomputed entropy will no longer be correct.
        Returns
        -------
        hmodel : bnpy HModel, with (possibly) some merged components
        SS : bnpy SuffStatBag, with (possibly) merged components
        evBound : correct ELBO for returned hmodel
                  guaranteed to be at least as large as input evBound    
    '''
    from MergeMove import run_many_merge_moves

    if self.hasMove('birth') and len(self.BirthCompIDs) > 0:
      compList = [x for x in self.BirthCompIDs] # need a copy
    else:
      compList = list()

    nMergeTrials = self.algParams['merge']['mergePerLap']

    hmodel, SS, newEvBound, MTracker = run_many_merge_moves(
                        hmodel, None, SS, evBound=evBound,
                        mPairIDs=mPairIDs, 
                        randstate=self.PRNG, nMergeTrials=nMergeTrials,
                        compList=compList, savedir=self.savedir,
                        **self.algParams['merge'])

    msg = 'MERGE: %3d/%3d accepted.' % (MTracker.nSuccess, MTracker.nTrial)
    if MTracker.nSuccess > 0:
      msg += ' ev improved + %.3e' % (newEvBound - evBound)
    if self.algParams['merge']['doVerbose'] >= 0:
      self.print_msg(msg)
    
    if self.algParams['merge']['doVerbose'] > 0:
      for msg in MTracker.InfoLog:
        self.print_msg(msg)

    # ------ Adjust indexing for counter that determines which comp to target
    if self.hasMove('birth'):
      for kA, kB in MTracker.acceptedIDs:
        self._adjustLapsSinceLastBirthForMerge(MTracker, kA, kB)
    # ------ Record accepted moves, so can adjust memoized stats later
    self.MergeLog = list()
    for kA, kB in MTracker.acceptedIDs:
      self.MergeLog.append(dict(kA=kA, kB=kB))
    # ------ Reset all precalculated merge terms
    if SS.hasMergeTerms():
      SS.setMergeFieldsToZero()

    return hmodel, SS, newEvBound

  def _adjustLapsSinceLastBirthForMerge(self, MTracker, kA, kB):
    ''' Adjust internal tracking of laps since birth
    '''
    compList = self.LapsSinceLastBirth.keys()
    newDict = defaultdict(int)
    for kk in compList:
      if kk == kA:
        newDict[kA] = np.maximum(self.LapsSinceLastBirth[kA], self.LapsSinceLastBirth[kB])
      elif kk < kB:
        newDict[kk] = self.LapsSinceLastBirth[kk]
      elif kk > kB:
        newDict[kk-1] = self.LapsSinceLastBirth[kk]
    self.LapsSinceLastBirth = newDict


  def run_merge_move_OLD(self, hmodel, Data, SS, evBound):
    ''' Run (potentially many) merge moves on hmodel,
          performing necessary bookkeeping to
            (1) avoid trying the same merge twice
            (2) avoid merging a component that has already been merged,
                since the precomputed entropy will no longer be correct.
        Returns
        -------
        hmodel : bnpy HModel, with (possibly) some merged components
        SS : bnpy SuffStatBag, with (possibly) merged components
        evBound : correct ELBO for returned hmodel
                  guaranteed to be at least as large as input evBound    
    '''
    import OldMergeMove
    self.MergeLog = list() # clear memory of recent merges!
    excludeList = list()
    excludePairs = defaultdict(lambda:set())
    nMergeAttempts = self.algParams['merge']['mergePerLap']
    trialID = 0
    while trialID < nMergeAttempts:

      # Synchronize contents of the excludeList and excludePairs
      # So that comp excluded in excludeList (due to accepted merge)
      #  is automatically contained in the set of excluded pairs 
      for kx in excludeList:
        for kk in excludePairs:
          excludePairs[kk].add(kx)
          excludePairs[kx].add(kk)

      for kk in excludePairs:
        if len(excludePairs[kk]) > hmodel.obsModel.K - 2:
          if kk not in excludeList:
            excludeList.append(kk)

      if len(excludeList) > hmodel.obsModel.K - 2:
        Log.info('Merge Done. No more options to try!')
        break # when we don't have any more comps to merge
        
      if self.hasMove('birth') and len(self.BirthCompIDs) > 0:
        kA = self.BirthCompIDs.pop()
        if kA in excludeList:
          continue
      else:
        kA = None

      hmodel, SS, evBound, MoveInfo = OldMergeMove.run_merge_move(
                 hmodel, None, SS, evBound, randstate=self.PRNG,
                 excludeList=excludeList, excludePairs=excludePairs,
                 kA=kA, **self.algParams['merge'])
      trialID += 1
      self.print_msg(MoveInfo['msg'])

      # Begin Bookkeeping!
      if 'kA' in MoveInfo and 'kB' in MoveInfo:
        kA = MoveInfo['kA']
        kB = MoveInfo['kB']
        excludePairs[kA].add(kB)
        excludePairs[kB].add(kA)

      if MoveInfo['didAccept']:
        self.MergeLog.append(dict(kA=kA, kB=kB))

        # Adjust excluded lists since components kB+1, kB+2, ... K
        #  have been shifted down by one due to removal of kB
        for kk in range(len(excludeList)):
          if excludeList[kk] > kB:
            excludeList[kk] -= 1

        # Exclude new merged component kA from future attempts        
        #  since precomputed entropy terms involving kA aren't good
        excludeList.append(kA)
    
        # Adjust excluded pairs to remove kB and shift down kB+1, ... K
        newExcludePairs = defaultdict(lambda:set())
        for kk in excludePairs.keys():
          ksarr = np.asarray(list(excludePairs[kk]))
          ksarr[ksarr > kB] -= 1
          if kk > kB:
            newExcludePairs[kk-1] = set(ksarr)
          elif kk < kB:
            newExcludePairs[kk] = set(ksarr)
        excludePairs = newExcludePairs

        # Adjust internal tracking of laps since birth
        if self.hasMove('birth'):
          compList = self.LapsSinceLastBirth.keys()
          newDict = defaultdict(int)
          for kk in compList:
            if kk == kA:
              newDict[kA] = np.maximum(self.LapsSinceLastBirth[kA], self.LapsSinceLastBirth[kB])
            elif kk < kB:
              newDict[kk] = self.LapsSinceLastBirth[kk]
            elif kk > kB:
              newDict[kk-1] = self.LapsSinceLastBirth[kk]
          self.LapsSinceLastBirth = newDict

    if SS.hasMergeTerms():
      SS.setMergeFieldsToZero()
    return hmodel, SS, evBound
  
