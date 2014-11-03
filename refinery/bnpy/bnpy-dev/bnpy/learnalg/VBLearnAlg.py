'''
VBLearnAlg.py

Implementation of both EM and VB for bnpy models

Notes
-------
Essentially, EM and VB are the same iterative *algorithm*,
repeating the steps of a monotonic increasing objective function until convergence.

EM recovers the parameters for a *point-estimate* of quantities of interest
while VB learns the parameters of an approximate *distribution* over quantities of interest

For more info, see the documentation [TODO]
'''
import numpy as np
from collections import defaultdict
from LearnAlg import LearnAlg

class VBLearnAlg( LearnAlg ):

  def __init__( self, **kwargs ):
    ''' Create VBLearnAlg, subtype of generic LearnAlg
    '''
    super(type(self), self).__init__( **kwargs )
    self.BirthLog = list()
    
  def fit(self, hmodel, Data):
    ''' Run EM/VB learning algorithm, fit global parameters of hmodel to Data
        Returns
        --------
        LP : local params from final pass of Data
        Info : dict of run information, with fields
              evBound : final ELBO evidence bound
              status : str message indicating reason for termination
                        {'converged', 'max passes exceeded'}
    '''
    prevBound = -np.inf
    LP = None
    mergeFlags = dict(doPrecompEntropy=True, doPrecompMergeEntropy=True)
    self.set_start_time_now()
    for iterid in xrange(self.algParams['nLap'] + 1):
      lap = self.algParams['startLap'] + iterid
      self.set_random_seed_at_lap(lap)

      # M step
      if iterid > 0:
        hmodel.update_global_params(SS) 
      
      if self.hasMove('birth') and iterid > 1:
        hmodel, LP = self.run_birth_move(hmodel, Data, SS, LP, iterid)
        
      # E step 
      LP = hmodel.calc_local_params(Data, LP, **self.algParamsLP)

      # Suff Stat step
      if self.hasMove('merge'):
        SS = hmodel.get_global_suff_stats(Data, LP, **mergeFlags)
      else:
        SS = hmodel.get_global_suff_stats(Data, LP)

      # ELBO calculation
      evBound = hmodel.calc_evidence(Data, SS, LP)

      # Attempt merge move      
      if self.hasMove('merge'):
        hmodel, SS, LP, evBound = self.run_merge_move(
                                          hmodel, Data, SS, LP, evBound)

      # Save and display progress
      self.add_nObs(Data.nObs)
      self.save_state(hmodel, iterid, lap, evBound)
      self.print_state(hmodel, iterid, lap, evBound)
      self.eval_custom_func(hmodel, iterid, float(iterid))
      # Check for Convergence!
      #  report warning if bound isn't increasing monotonically
      isConverged = self.verify_evidence( evBound, prevBound )
      if isConverged:
        break
      prevBound = evBound

    #Finally, save, print and exit
    if isConverged:
      status = "converged."
    else:
      status = "max passes thru data exceeded."
    self.save_state(hmodel,iterid, lap, evBound, doFinal=True)    
    self.print_state(hmodel,iterid, lap, evBound, doFinal=True, status=status)

    return LP, self.buildRunInfo(evBound, status, nLap=lap)


  ########################################################### Birth Move
  ###########################################################
  def run_birth_move(self, hmodel, Data, SS, LP, lap):
    ''' Run birth move on hmodel
    ''' 
    import BirthMove # avoid circular import
    self.BirthLog = list()
    if not self.do_birth_at_lap(lap):
      return hmodel, LP
      
    kbirth = BirthMove.select_birth_component(SS, 
                          randstate=self.PRNG,
                          **self.algParams['birth'])

    TargetData = BirthMove.subsample_data(Data, LP, kbirth, 
                          randstate=self.PRNG,
                          **self.algParams['birth'])

    hmodel, SS, MoveInfo = BirthMove.run_birth_move(
                 hmodel, TargetData, SS, ktarget=kbirth, randstate=self.PRNG, 
                 **self.algParams['birth'])
    self.print_msg(MoveInfo['msg'])
    self.BirthLog.extend(MoveInfo['birthCompIDs'])
    LP = None
    return hmodel, LP
    

  ########################################################### Merge Move
  ###########################################################
  def run_merge_move(self, hmodel, Data, SS, LP, evBound):
    ''' Run merge move on hmodel
    ''' 
    import MergeMove
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
        break # when we don't have any more comps to merge
        
      if len(self.BirthLog) > 0:
        kA = self.BirthLog.pop()
        if kA in excludeList:
          continue
      else:
        kA = None

      oldEv = hmodel.calc_evidence(SS=SS)
      hmodel, SS, evBound, MoveInfo = MergeMove.run_merge_move(
                 hmodel, Data, SS, evBound, kA=kA, randstate=self.PRNG,
                 excludeList=excludeList, excludePairs=excludePairs,
                  **self.algParams['merge'])
      newEv = hmodel.calc_evidence(SS=SS)
      
      trialID += 1
      self.print_msg(MoveInfo['msg'])
      if 'kA' in MoveInfo and 'kB' in MoveInfo:
        kA = MoveInfo['kA']
        kB = MoveInfo['kB']
        excludePairs[kA].add(kB)
        excludePairs[kB].add(kA)

      if MoveInfo['didAccept']:
        assert newEv > oldEv
        kA = MoveInfo['kA']
        kB = MoveInfo['kB']
        # Adjust excludeList since components kB+1, kB+2, ... K
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

        # Update LP to reflect this merge!
        LPkeys = LP.keys()
        keepLPkeys = hmodel.allocModel.get_keys_for_memoized_local_params()

        for key in LPkeys:
          if key in keepLPkeys:
            LP[key][:, kA] = LP[key][:, kA] + LP[key][:, kB]
            LP[key] = np.delete(LP[key], kB, axis=1)
    return hmodel, SS, LP, evBound



