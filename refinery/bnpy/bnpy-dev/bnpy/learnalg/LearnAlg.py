'''
LearnAlg.py
Abstract base class for learning algorithms for HModel models

Defines some generic routines for
  * saving global parameters
  * assessing convergence
  * printing progress updates to stdout
  * recording run-time
'''
from bnpy.ioutil import ModelWriter
from bnpy.util import closeAtMSigFigs, isEvenlyDivisibleFloat
import numpy as np
import time
import os
import logging
import scipy.io

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

class LearnAlg(object):

  def __init__(self, savedir=None, seed=0, 
                     algParams=dict(), outputParams=dict(),
                     onLapCompleteFunc=lambda:None, onFinishFunc=lambda:None,
               ):
    ''' Constructs and returns a LearnAlg object
    ''' 
    if type(savedir) == str:
      self.savedir = os.path.splitext(savedir)[0]
    else:
      self.savedir = None
    self.seed = int(seed)
    self.PRNG = np.random.RandomState(self.seed)
    self.algParams = algParams
    self.outputParams = outputParams
    self.TraceLaps = set()
    self.evTrace = list()
    self.SavedIters = set()
    self.PrintIters = set()
    self.nObsProcessed = 0
    self.algParamsLP = dict()
    for k,v in algParams.items():
      if k.count('LP') > 0:
        self.algParamsLP[k] = v
    
  def fit(self, hmodel, Data):
    ''' Execute learning algorithm for hmodel on Data
        This method is extended by any subclass of LearnAlg

        Returns
        -------
        LP : local params dictionary of resulting model
    '''
    pass


  def set_random_seed_at_lap(self, lap):
    ''' Set internal random generator deterministically
          based on provided seed (unique to this run) and 
          the number of passes thru the data,
          so we can reproduce runs without starting over
    '''
    if isEvenlyDivisibleFloat(lap, 1.0):
      self.PRNG = np.random.RandomState(self.seed + int(lap))
    
  def set_start_time_now(self):
    ''' Record start time (in seconds since 1970)
    '''
    self.start_time = time.time()    

  def add_nObs(self, nObs):
    ''' Update internal count of total number of data observations processed.
        Each lap thru dataset of size N, this should be updated by N
    '''
    self.nObsProcessed += nObs

  def get_elapsed_time(self):
    ''' Returns float of elapsed time (in seconds) since this object's
        set_start_time_now() method was called
    '''
    return time.time() - self.start_time

  def buildRunInfo(self, evBound, status, nLap=None):
    ''' Create dict of information about the current run
    '''
    return dict(evBound=evBound, status=status, nLap=nLap,
                evTrace=self.evTrace, lapTrace=self.TraceLaps)

  ##################################################### Fcns for birth/merges
  ##################################################### 
  def hasMove(self, moveName):
    if moveName in self.algParams:
      return True
    return False

  ##################################################### Verify evidence
  #####################################################  grows monotonically
  def verify_evidence(self, evBound=0.00001, prevBound=0, lapFrac=None):
    ''' Compare current and previous evidence (ELBO) values,
        verify that (within numerical tolerance) increases monotonically
    '''
    if np.isnan(evBound):
      raise ValueError("Evidence should never be NaN")
    if np.isinf(prevBound):
      return False
    isIncreasing = prevBound <= evBound
    M = self.algParams['convergeSigFig']
    isWithinTHR = closeAtMSigFigs(prevBound, evBound, M=M)
    mLPkey = 'doMemoizeLocalParams'
    if not isIncreasing and not isWithinTHR:
      serious = True
      if self.hasMove('birth') \
         and (len(self.BirthCompIDs) > 0 or len(self.ModifiedCompIDs) > 0):
        warnMsg = 'ev decreased during a birth'
        warnMsg += ' (so monotonic increase not guaranteed)\n'
        serious = False
      elif mLPkey in self.algParams and not self.algParams[mLPkey]:
        warnMsg = 'ev decreased when doMemoizeLocalParams=0'
        warnMsg += ' (so monotonic increase not guaranteed)\n'
        serious = False
      else:
        warnMsg = 'evidence decreased!\n'
      warnMsg += '    prev = % .15e\n' % (prevBound)
      warnMsg += '     cur = % .15e\n' % (evBound)
      if lapFrac is None:
        prefix = "WARNING: "
      else:
        prefix = "WARNING @ %.3f: " % (lapFrac)

      if serious or not self.algParams['doShowSeriousWarningsOnly']:
        Log.error(prefix + warnMsg)
    return isWithinTHR 


  #########################################################  Save to file
  #########################################################  
  def save_state(self, hmodel, iterid, lap, evBound, doFinal=False):
    ''' Save state of the hmodel's global parameters and evBound
    '''  
    traceEvery = self.outputParams['traceEvery']
    if traceEvery <= 0:
      traceEvery = -1
    doTrace = isEvenlyDivisibleFloat(lap, traceEvery) or iterid < 3
    
    if traceEvery > 0 and (doFinal or doTrace) and lap not in self.TraceLaps:
      # Record current evidence
      self.evTrace.append(evBound)
      self.TraceLaps.add(lap)

      # Exit here if we're not saving to disk
      if self.savedir is None:
        return
    
      # Record current state to plain-text files
      with open( self.mkfile('laps.txt'), 'a') as f:        
        f.write('%.4f\n' % (lap))
      with open( self.mkfile('evidence.txt'), 'a') as f:        
        f.write('%.9e\n' % (evBound))
      with open( self.mkfile('nObs.txt'), 'a') as f:
        f.write('%d\n' % (self.nObsProcessed))
      with open( self.mkfile('times.txt'), 'a') as f:
        f.write('%.3f\n' % (self.get_elapsed_time()))
      if self.hasMove('birth') or self.hasMove('merge'):
        with open( self.mkfile('K.txt'), 'a') as f:
          f.write('%d\n' % (hmodel.obsModel.K))

    saveEvery = self.outputParams['saveEvery']
    if saveEvery <= 0 or self.savedir is None:
      return

    doSave = isEvenlyDivisibleFloat(lap, saveEvery) or iterid < 3
    if (doFinal or doSave) and iterid not in self.SavedIters:
      self.SavedIters.add(iterid)
      with open(self.mkfile('laps-saved-params.txt'), 'a') as f:        
        f.write('%.4f\n' % (lap))
      prefix = ModelWriter.makePrefixForLap(lap)
      ModelWriter.save_model(hmodel, self.savedir, prefix,
                              doSavePriorInfo=(iterid<1), doLinkBest=True)

  # Define temporary function that creates files in this alg's output dir
  def mkfile(self, fname):
    return os.path.join(self.savedir, fname)

  def getFileWriteMode(self):
    if self.savedir is None:
      return None
    return 'a'
    
  ######################################################### Plot Results
  ######################################################### 
  def plot_results(self, hmodel, Data, LP):
    ''' Plot learned model parameters
    '''
    pass

  #########################################################  Print State
  #########################################################  
  def print_state(self, hmodel, iterid, lap, evBound, doFinal=False, status='', rho=None):
    printEvery = self.outputParams['printEvery']
    if printEvery <= 0:
      return None
    doPrint = iterid < 3 or isEvenlyDivisibleFloat(lap, printEvery)
  
    if rho is None:
      rhoStr = ''
    else:
      rhoStr = '%.4f |' % (rho)

    if iterid == lap:
      lapStr = '%7d' % (lap)
    else:
      lapStr = '%7.3f' % (lap)

    maxLapStr = '%d' % (self.algParams['nLap'] + self.algParams['startLap'])
    
    logmsg = '  %s/%s after %6.0f sec. | K %4d | ev % .9e %s'
    # Print asterisk for early iterations of memoized,
    #  before the method has made one full pass thru data
    if self.__class__.__name__.count('Memo') > 0:
      if lap < self.algParams['startLap'] + 1.0:
        logmsg = '  %s/%s after %6.0f sec. | K %4d |*ev % .9e %s'

    logmsg = logmsg % (lapStr, 
                        maxLapStr,
                        self.get_elapsed_time(),
                        hmodel.allocModel.K,
                        evBound, 
                        rhoStr)

    if (doFinal or doPrint) and iterid not in self.PrintIters:
      self.PrintIters.add(iterid)
      Log.info(logmsg)
    if doFinal:
      Log.info('... done. %s' % (status))
      
  def print_msg(self, msg):
      ''' Prints a string msg to stdout,
            without needing to import logging method into subclass. 
      '''
      Log.info(msg)

  #########################################################
  def isFirstBatch(self, lapFrac):
    ''' Returns True/False for whether given batch is last (for current lap)
    '''
    if self.lapFracInc == 1.0: # Special case, nBatch == 1
      isFirstBatch = True
    else:
      isFirstBatch = np.allclose(lapFrac - np.floor(lapFrac), self.lapFracInc)
    return isFirstBatch

  def isLastBatch(self, lapFrac):
    ''' Returns True/False for whether given batch is last (for current lap)
    '''
    return lapFrac % 1 == 0

  def do_birth_at_lap(self, lapFrac):
    ''' Returns True/False for whether birth happens at given lap
    '''
    if 'birth' not in self.algParams:
      return False
    nLapTotal = self.algParams['nLap']
    frac = self.algParams['birth']['fracLapsBirth']
    if lapFrac > nLapTotal:
      return False
    return (nLapTotal <= 5) or (lapFrac <= np.ceil(frac * nLapTotal))

  def eval_custom_func(self, hmodel, iterid, lapFrac):
      ''' Evaluates a custom hook function called customFunc.py in the path specified in algParams['customFuncPath']
      '''
      import ast
      customFuncPath = self.algParams['customFuncPath']
      customFuncArgs = self.algParams['customFuncArgs']
      nLapTotal = self.algParams['nLap']
      percentDone = lapFrac/nLapTotal
      if customFuncPath is not None and customFuncPath != 'None':
        import sys
        sys.path.append(customFuncPath)
        import customFunc
        if lapFrac % 1 != 0:
            customFunc.onBatchComplete(hmodel, percentDone , customFuncArgs)
        elif lapFrac % 1 == 0 and lapFrac < nLapTotal:
            customFunc.onLapComplete(hmodel, percentDone , customFuncArgs)
        else:
            customFunc.onAlgorithmComplete(hmodel, percentDone , customFuncArgs)
