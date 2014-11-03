'''
StochasticOnlineVBLearnAlg.py

Implementation of stochastic online VB (soVB) for bnpy models
'''
import numpy as np
from LearnAlg import LearnAlg

class StochasticOnlineVBLearnAlg(LearnAlg):

  def __init__(self, **kwargs):
    ''' Creates stochastic online learning algorithm, 
        with fields rhodelay, rhoexp that define learning rate schedule.
    '''
    super(type(self),self).__init__(**kwargs)
    self.rhodelay = self.algParams['rhodelay']
    self.rhoexp = self.algParams['rhoexp']

  def fit(self, hmodel, DataIterator, SS=None):
    ''' Run soVB learning algorithm, fit global parameters of hmodel to Data
        Returns
        --------
        LP : local params from final pass of Data
        Info : dict of run information, with fields
              evBound : final ELBO evidence bound
              status : str message indicating reason for termination
                        {'all data processed'}
    '''
    LP = None
    rho = 1.0 # Learning rate
    nBatch = float(DataIterator.nBatch)

    # Set-up progress-tracking variables
    iterid = -1
    lapFrac = np.maximum(0, self.algParams['startLap'] - 1.0/nBatch)
    if lapFrac > 0:
      # When restarting an existing run,
      #  need to start with last update for final batch from previous lap
      DataIterator.lapID = int(np.ceil(lapFrac)) - 1
      DataIterator.curLapPos = nBatch - 2
      iterid = int(nBatch * lapFrac) - 1

    self.set_start_time_now()
    while DataIterator.has_next_batch():

      # Grab new data
      Dchunk = DataIterator.get_next_batch()

      # Update progress-tracking variables
      iterid += 1
      lapFrac += 1.0/nBatch
      self.set_random_seed_at_lap(lapFrac)

      # M step with learning rate
      if SS is not None:
        rho = (iterid + self.rhodelay) ** (-1.0 * self.rhoexp)
        hmodel.update_global_params(SS, rho)
      
      # E step
      LP = hmodel.calc_local_params(Dchunk)
      SS = hmodel.get_global_suff_stats(Dchunk, LP, doAmplify=True)

      # ELBO calculation
      evBound = hmodel.calc_evidence(Dchunk, SS, LP)      

      # Save and display progress
      self.add_nObs(Dchunk.nObs)
      self.save_state(hmodel, iterid, lapFrac, evBound)
      self.print_state(hmodel, iterid, lapFrac, evBound)
    
    #Finally, save, print and exit
    status = "all data processed."
    self.save_state(hmodel,iterid, lapFrac, evBound, doFinal=True)    
    self.print_state(hmodel, iterid, lapFrac, evBound, doFinal=True, status=status)
    return None, self.buildRunInfo(evBound, status)
