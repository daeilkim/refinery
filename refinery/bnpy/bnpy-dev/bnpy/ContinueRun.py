'''
 User-facing executable script for contining saved runs from Run.py
    
  Quickstart (Command Line)
  -------
  To continue a run associated with data, model, algorithm, and job name
  for 10 more laps, do

  $ python -m bnpy.ContinueRun <Data> <AllocModel> <ObsModel> <AlgName> <job> --nLap 10
  
  Usage
  -------
  TODO: write better doc
'''
import os
import sys
import logging
import numpy as np
import bnpy
import argparse
BNPYArgParser = bnpy.ioutil.BNPYArgParser

Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

FullDataAlgSet = ['EM','VB']
OnlineDataAlgSet = ['soVB', 'moVB']

def continueRun(dataName=None, allocModelName=None, obsModelName=None, 
        algName=None,
        doSaveToDisk=True, doWriteStdOut=True, **kwargs):
  ''' Fit specified model to data with learning algorithm.
    
      Usage
      -------
      TODO

      Args
      -------
      dataName : either one of
                  * bnpy Data object,
                  * string filesystem path of a Data module within BNPYDATADIR
      allocModelName : string name of allocation (latent structure) model
                        {MixModel, DPMixModel, AdmixModel, HMM, etc.}
      obsModelName : string name of observation (likelihood) model
                        {Gauss, ZMGauss, WordCount, etc.}
      algName : string name of algorithm
                        {EM, VB, moVB, soVB}
      Returns
      -------
      hmodel : best model fit to the dataset (across nTask runs)
      LP : local parameters of that best model on the dataset
  '''
  hasReqArgs = dataName is not None
  hasReqArgs &= allocModelName is not None
  hasReqArgs &= obsModelName is not None
  hasReqArgs &= algName is not None
  
  if hasReqArgs:
    ReqArgs = dict(dataName=dataName, allocModelName=allocModelName, \
                    obsModelName=obsModelName, algName=algName)
  else:
    ReqArgs = BNPYArgParser.parseRequiredArgs()
    dataName = ReqArgs['dataName']
    allocModelName = ReqArgs['allocModelName']
    obsModelName = ReqArgs['obsModelName']
    algName = ReqArgs['algName']

  parser = argparse.ArgumentParser()
  parser.add_argument('--jobname', default='defaultjob')
  parser.add_argument('--taskid', default=1, type=int)
  parser.add_argument('--startLap', default=1.0, type=float)
  parser.add_argument('--nLap', default=10, type=int)

  argDict, ignored = BNPYArgParser.applyParserToKwArgDict(parser, **kwargs)
  hmodel, LP, Info = _continue_task_internal(
                      argDict['jobname'], argDict['taskid'],
                      argDict['startLap'], argDict['nLap'],
                      dataName, allocModelName, obsModelName, algName,
                      ReqArgs, doSaveToDisk, doWriteStdOut)
  return hmodel, LP, Info

########################################################### RUN SINGLE TASK 
###########################################################
def _continue_task_internal(jobname, taskid, startLap, nLap, \
                      dataName, allocModelName, obsModelName, algName, \
                      ReqArgs, doSaveToDisk, doWriteStdOut):
  ''' Internal method (should never be called by end-user!)
      Executes learning for a particular job and particular taskid.
      
      Returns
      -------
        hmodel : bnpy HModel, fit to the data
        LP : Local parameter (LP) dict for the specific dataset
        RunInfo : dict of information about the run, with fields
                'evBound' log evidence for hmodel on the specified dataset
                'evTrace' vector of evBound at every traceEvery laps
  '''
  algseed = createUniqueRandomSeed(jobname, taskID=taskid)
  dataorderseed = createUniqueRandomSeed('', taskID=taskid)

  taskoutpath = getOutputPath(ReqArgs, jobname, taskID=taskid)
  configLoggingToConsoleAndFile(taskoutpath, doSaveToDisk, doWriteStdOut)

  # Create model from saved parameters
  ModelReader = bnpy.ioutil.ModelReader
  hmodel, startLap = ModelReader.loadModelForLap(taskoutpath, startLap)

  KwArgs, UnkArgs = readArgsFromFile(taskoutpath)
  if algName in FullDataAlgSet:
    KwArgs[algName]['nLap'] = nLap
  else:
    KwArgs['OnlineDataPrefs']['nLap'] = nLap
    KwArgs['OnlineDataPrefs']['startLap'] = startLap
  if type(dataName) is str:
    Data, InitData = loadData(ReqArgs, KwArgs, UnkArgs, dataorderseed)
  else:
    Data = dataName
    InitData = dataName
    if algName in OnlineDataAlgSet:
      Data = Data.to_minibatch_iterator(dataorderseed=dataorderseed,
                                        **KwArgs['OnlineDataPrefs'])

  # Create learning algorithm
  KwArgs[algName]['startLap'] = startLap
  learnAlg = createLearnAlg(Data, hmodel, ReqArgs, KwArgs, 
                              algseed=algseed, savepath=taskoutpath)

  # Write descriptions to the log
  if taskid == 1:
    Log.info(Data.get_text_summary())
    Log.info(Data.summarize_num_observations())
    Log.info(hmodel.get_model_info())
    Log.info('Learn Alg: %s. startLap %d' % (algName, startLap))    
  Log.info('Trial %2d | alg. seed: %d | data order seed: %d' \
               % (taskid, algseed, dataorderseed))
  Log.info('savepath: %s' % (taskoutpath))

  # Fit the model to the data!
  LP, RunInfo = learnAlg.fit(hmodel, Data)
  return hmodel, LP, RunInfo
  

########################################################### Load Data
###########################################################
def loadData(ReqArgs, KwArgs, DataArgs, dataorderseed):
  ''' Load DataObj specified by the user, using particular random seed.
      Returns
      --------
      either 
        Data, InitData  
      or
        DataIterator, InitData

      InitData must be a bnpy.data.DataObj object.
      This DataObj is used for two early-stage steps in the training process
        (a) Constructing observation model so that it has appropriate dimensions
            For example, with 3D real data,
            can only model the observations with a Gaussian over 3D vectors. 
        (b) Initializing global model parameters
            Esp. in online settings, avoiding local optima might require using parameters
            that are initialized from a much bigger dataset than each individual batch.
      For most full dataset learning scenarios, InitData can be the same as Data.
  '''
  sys.path.append(os.environ['BNPYDATADIR'])
  datamod = __import__(ReqArgs['dataName'],fromlist=[])
  algName = ReqArgs['algName']
  if algName in FullDataAlgSet:
    Data = datamod.get_data(**DataArgs)
    return Data, Data
  elif algName in OnlineDataAlgSet:
    KwArgs[algName]['nLap'] = KwArgs['OnlineDataPrefs']['nLap']
    InitData = datamod.get_data(**DataArgs)
    OnlineDataArgs = KwArgs['OnlineDataPrefs']
    OnlineDataArgs['dataorderseed'] = dataorderseed
    OnlineDataArgs.update(DataArgs)
    DataIterator = datamod.get_minibatch_iterator(**OnlineDataArgs)
    return DataIterator, InitData
  
########################################################### Create Model
###########################################################
def createModel(Data, ReqArgs, KwArgs):
  ''' Creates a bnpy HModel object for the given Data
      This object is responsible for:
       * storing global parameters
       * providing methods to perform model-specific subroutines for learning,
          such as calc_local_params (E-step) or get_global_suff_stats
      Returns
      -------
      hmodel : bnpy.HModel object, whose allocModel is of type ReqArgs['allocModelName']
                                    and obsModel is of type ReqArgs['obsModelName']
               This model has fully defined prior distribution parameters,
                 but *will not* have initialized global parameters.
               It must be initialized via hmodel.init_global_params(...) before use.
  '''
  algName = ReqArgs['algName']
  aName = ReqArgs['allocModelName']
  oName = ReqArgs['obsModelName']
  aPriorDict = KwArgs[aName]
  oPriorDict = KwArgs[oName]
  hmodel = bnpy.HModel.CreateEntireModel(algName, aName, oName, aPriorDict, oPriorDict, Data)
  return hmodel  


########################################################### Create LearnAlg
###########################################################
def createLearnAlg(Data, model, ReqArgs, KwArgs, 
                    algseed=0, savepath=None):
  ''' Creates a bnpy LearnAlg object for the given Data and model
      This object is responsible for:
        * preparing a directory to save the data (savepath)
        * setting appropriate random seeds specific to the *learning algorithm*
          
    Returns
    -------
    learnAlg : LearnAlg [or subclass] object
               type defined by ArgDict['algName'], one of {EM, VB, soVB, moVB}
  '''
  algName = ReqArgs['algName']
  algP = KwArgs[algName]
  if 'birth' in KwArgs:
    algP['birth'] = KwArgs['birth']
  if 'merge' in KwArgs:
    algP['merge'] = KwArgs['merge']
  outputP = KwArgs['OutputPrefs']
  if algName == 'EM' or algName == 'VB':
    learnAlg = bnpy.learnalg.VBLearnAlg(savedir=savepath, seed=algseed, \
                                      algParams=algP, outputParams=outputP)
  elif algName == 'soVB':
    learnAlg = bnpy.learnalg.StochasticOnlineVBLearnAlg(savedir=savepath, seed=algseed, algParams=algP, outputParams=outputP)
  elif algName == 'moVB':
    learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=savepath, seed=algseed, algParams=algP, outputParams=outputP)
  else:
    raise NotImplementedError("Unknown learning algorithm " + algName)
  return learnAlg


########################################################### Write Args to File
###########################################################
def readArgsFromFile(taskoutpath):
  ''' Read args as key/val pairs from plain-text file
      so that we can figure out what settings were used for a saved run
  '''
  import json
  import glob
  KwArgs = dict()
  for argfilepath in glob.glob(os.path.join(taskoutpath,'args-*.txt')):
    # Input: fullpath like /path/to/my/save/dir/args-VB.txt
    # Output: extract the "VB" part
    pathparts = argfilepath.split(os.path.sep)
    curKey = pathparts[-1].split('-')[1].split('.txt')[0]
    with open(argfilepath, 'r') as f:
      curDict = json.load(f)
    KwArgs[curKey] = curDict
  return KwArgs, dict()

########################################################### Config Subroutines
###########################################################
def createUniqueRandomSeed( jobname, taskID=0):
  ''' Get unique seed for a random number generator,
       deterministically using the jobname and taskID.
      This seed is reproducible on any machine, regardless of OS or 32/64 arch.
      Returns
      -------
      seed : integer seed for a random number generator,
                such as numpy's RandomState object.
  '''
  import hashlib
  if jobname.count('-') > 0:
    jobname = jobname.split('-')[0]
  if len(jobname) > 5:
    jobname = jobname[:5]
  
  seed = int( hashlib.md5( jobname+str(taskID) ).hexdigest(), 16) % 1e7
  return int(seed)
  
  
def getOutputPath(ReqArgs, jobname, taskID=0 ):
  ''' Get a valid file system path for writing output from learning alg execution.
      Returns
      --------
      outpath : absolute path to a directory on this file system.
                Note: this directory may not exist yet.
  '''
  dataName = ReqArgs['dataName']
  if type(dataName) is not str:
    dataName = dataName.get_short_name()
  return os.path.join(os.environ['BNPYOUTDIR'], 
                       dataName, 
                       ReqArgs['allocModelName'],
                       ReqArgs['obsModelName'],
                       ReqArgs['algName'],
                       jobname, 
                       str(taskID) )


def configLoggingToConsoleAndFile(taskoutpath, doSaveToDisk=True, doWriteStdOut=True):
  Log.handlers = [] # remove pre-existing handlers!
  formatter = logging.Formatter('%(message)s')
  ###### Config logger to save a transcript of log messages to plain-text file  
  if doSaveToDisk:
    fh = logging.FileHandler(os.path.join(taskoutpath,"transcript.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    Log.addHandler(fh)
  ###### Config logger that can write to stdout
  if doWriteStdOut:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    Log.addHandler(ch)
  ##### Config a null logger to avoid error messages about no handler existing
  if not doSaveToDisk and not doWriteStdOut:
    Log.addHandler(logging.NullHandler())


if __name__ == '__main__':
  continueRun()
