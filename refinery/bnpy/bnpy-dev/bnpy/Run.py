'''
 User-facing executable script for running experiments that
  train bnpy models using a variety of possible inference algorithms
    ** Expectation Maximization (EM)
    ** Variational Bayesian Inference (VB)
    ** Stochastic Online Variational Bayesian Inference (soVB)
    ** Memoized Online Variational Bayesian Inference (moVB)
    
  Quickstart (Command Line)
  -------
  To run EM for a 3-component GMM on easy, predefined toy data, do
  $ python -m bnpy.Run AsteriskK8 MixModel Gauss EM --K=3

  Quickstart (within python script)
  --------
  To do the same as above, just call the run method:
  >> hmodel = run('AsteriskK8', 'MixModel', 'Gauss', 'EM', K=3)
  
  Usage
  -------
  TODO: write better doc
'''
import os
import sys
import logging
import numpy as np
import bnpy
from bnpy.ioutil import BNPYArgParser

# Configure Logger
Log = logging.getLogger('bnpy')
Log.setLevel(logging.DEBUG)

FullDataAlgSet = ['EM','VB']
OnlineDataAlgSet = ['soVB', 'moVB']

def run(dataName=None, allocModelName=None, obsModelName=None, algName=None, \
                      doSaveToDisk=True, doWriteStdOut=True, 
                      taskID=None, **kwargs):
  ''' Fit specified model to data with learning algorithm.
    
      Usage
      -------
      To fit a Gauss MixModel to a custom dataset defined in matrix X 
      >> Data = bnpy.data.XData(X)
      >> hmodel = run(Data, 'MixModel', 'Gauss', 'EM', K=3, nLap=10)

      To load a dataset specified in a specific script
      For example, 2D toy data in demodata/AsteriskK8.py
      >> hmodel = run('AsteriskK8', 'MixModel', 'Gauss', 'VB', K=3)
      
      To run 5 tasks (separate initializations) and get best of 5 runs:
      >> opts = dict(K=8, nLap=100, printEvery=0)
      >> hmodel = run('AsteriskK8','MixModel','Gauss','VB', nTask=5, **opts)

      Args
      -------
      dataName : either one of
                  * bnpy Data object,
                  * string filesystem path of Data module within BNPYDATADIR
      allocModelName : string name of allocation (latent structure) model
                        {MixModel, DPMixModel, AdmixModel, HMM, etc.}
      obsModelName : string name of observation (likelihood) model
                        {Gauss, ZMGauss, WordCount, etc.}
      **kwargs : keyword args defining properties of the model or alg
                  see Doc for details [TODO]
      Returns
      -------
      hmodel : best model fit to the dataset (across nTask runs)
      LP : local parameters of that best model on the dataset
      evBound : log evidence (ELBO) for the best model on the dataset
                  scalar, real value where larger value implies better model
  '''
  hasReqArgs = dataName is not None
  hasReqArgs &= allocModelName is not None
  hasReqArgs &= obsModelName is not None
  hasReqArgs &= algName is not None
  
  if hasReqArgs:
    ReqArgs = dict(dataName=dataName, allocModelName=allocModelName,
                    obsModelName=obsModelName, algName=algName)
  else:
    ReqArgs = BNPYArgParser.parseRequiredArgs()
    dataName = ReqArgs['dataName']
    allocModelName = ReqArgs['allocModelName']
    obsModelName = ReqArgs['obsModelName']
    algName = ReqArgs['algName']
  KwArgs, UnkArgs = BNPYArgParser.parseKeywordArgs(ReqArgs, **kwargs)
  
  jobname = KwArgs['OutputPrefs']['jobname']

  if taskID is None:
    starttaskid = KwArgs['OutputPrefs']['taskid']
  else:
    starttaskid = taskID
    KwArgs['OutputPrefs']['taskid'] = taskID
  nTask = KwArgs['OutputPrefs']['nTask']
  
  bestInfo = None
  bestEvBound = -np.inf
  for taskid in range(starttaskid, starttaskid + nTask):
    hmodel, LP, Info = _run_task_internal(jobname, taskid, nTask,
                      ReqArgs, KwArgs, UnkArgs,
                      dataName, allocModelName, obsModelName, algName,
                      doSaveToDisk, doWriteStdOut)
    if (Info['evBound'] > bestEvBound):
      bestModel = hmodel
      bestLP = LP
      bestEvBound = Info['evBound']
      bestInfo = Info
  return bestModel, bestLP, bestInfo

########################################################### RUN SINGLE TASK 
###########################################################
def _run_task_internal(jobname, taskid, nTask,
                      ReqArgs, KwArgs, UnkArgs,
                      dataName, allocModelName, obsModelName, algName,
                      doSaveToDisk, doWriteStdOut):
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

  if doSaveToDisk:
    taskoutpath = getOutputPath(ReqArgs, KwArgs, taskID=taskid)
    createEmptyOutputPathOnDisk(taskoutpath)
    writeArgsToFile(ReqArgs, KwArgs, taskoutpath)
  else:
    taskoutpath = None
  configLoggingToConsoleAndFile(taskoutpath, doSaveToDisk, doWriteStdOut)
  
  if type(dataName) is str:   
    Data, InitData = loadData(ReqArgs, KwArgs, UnkArgs, dataorderseed)
  else:
    Data = dataName
    InitData = dataName
    assert str(type(InitData)).count('bnpy.data') > 0
    if algName in OnlineDataAlgSet:
      KwArgs[algName]['nLap'] = KwArgs['OnlineDataPrefs']['nLap']
      OnlineDataArgs = KwArgs['OnlineDataPrefs']
      OnlineDataArgs['dataorderseed'] = dataorderseed
      OnlineDataArgs.update(UnkArgs) # add custom args
      Data = Data.to_minibatch_iterator(**OnlineDataArgs)

  # Create and initialize model parameters
  hmodel = createModel(InitData, ReqArgs, KwArgs)
  hmodel.init_global_params(InitData, seed=algseed,
                            **KwArgs['Initialization'])

  # Create learning algorithm
  learnAlg = createLearnAlg(Data, hmodel, ReqArgs, KwArgs,
                              algseed=algseed, savepath=taskoutpath)

  # Check if running on grid
  try:
    jobID = int(os.getenv('JOB_ID'))
  except TypeError:
    jobID = 0
  if jobID > 0:
    Log.info('SGE Grid Job ID: %d' % (jobID))
    # Create symlinks of the captured stdout, stdout and log in bnpy output directory
    #  so everything is in the same place
    os.symlink(os.getenv('SGE_STDOUT_PATH'), os.path.join(taskoutpath, 'stdout.log'))
    os.symlink(os.getenv('SGE_STDERR_PATH'), os.path.join(taskoutpath, 'stderr.log'))

  # Write descriptions to the log
  if taskid == 1 or jobID > 0:
    Log.info(Data.get_text_summary())
    Log.info(Data.summarize_num_observations())
    if type(Data) != type(InitData):
      Log.info(InitData.get_text_summary(doCommon=False))
    Log.info(hmodel.get_model_info())
    Log.info('Learn Alg: %s' % (algName))

  Log.info('Trial %2d/%d | alg. seed: %d | data order seed: %d' \
               % (taskid, nTask, algseed, dataorderseed))
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
        (a) Constructing observation model of appropriate dimensions
            For example, with 3D real data,
            can only model the observations with a Gaussian over 3D vectors. 
        (b) Initializing global model parameters
            Esp. in online settings, avoiding local optima might require
             using parameters initialized from a much bigger dataset
             than each individual batch.
      For full dataset learning scenarios, InitData can be the same as Data.
  '''
  datamod = __import__(ReqArgs['dataName'],fromlist=[])
  algName = ReqArgs['algName']
  if algName in FullDataAlgSet:
    Data = datamod.get_data(**DataArgs)
    return Data, Data
  elif algName in OnlineDataAlgSet:
    InitData = datamod.get_data(**DataArgs)

    if 'OnlineDataPrefs' in KwArgs:
      KwArgs[algName]['nLap'] = KwArgs['OnlineDataPrefs']['nLap']
      OnlineDataArgs = KwArgs['OnlineDataPrefs']
      OnlineDataArgs['dataorderseed'] = dataorderseed
      OnlineDataArgs.update(DataArgs)
      DataIterator = datamod.get_minibatch_iterator(**OnlineDataArgs)
    else:
      DataIterator = None
    return DataIterator, InitData
  
########################################################### Create Model
###########################################################
def createModel(Data, ReqArgs, KwArgs):
  ''' Creates a bnpy HModel object for the given Data
      This object is responsible for:
       * storing global parameters
       * methods to perform model-specific subroutines for learning,
          such as calc_local_params (E-step) or get_global_suff_stats
      Returns
      -------
      hmodel : bnpy.HModel object, 
                 allocModel is of type ReqArgs['allocModelName']
                 obsModel is of type ReqArgs['obsModelName']
               This model has fully defined prior parameters,
                 but *will not* have initialized global parameters.
              Need to run hmodel.init_global_params() before use.
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
def createLearnAlg(Data, model, ReqArgs, KwArgs, algseed=0, savepath=None):
  ''' Creates a bnpy LearnAlg object for the given Data and model
      This object is responsible for:
        * preparing a directory to save the data (savepath)
        * setting random seeds specific to the *learning algorithm*
          
    Returns
    -------
    learnAlg : LearnAlg [or subclass] object
               type is defined by string in ReqArgs['algName']
                one of {'EM', 'VB', 'soVB', 'moVB'}
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
def writeArgsToFile( ReqArgs, KwArgs, taskoutpath ):
  ''' Save arguments as key/val pairs to a plain text file
      so that we inspect settings for a saved run later on
  '''
  import json
  ArgDict = ReqArgs
  ArgDict.update(KwArgs)
  #RelevantOpts = dict(Initialization=1, OutputPrefs=1, OnlineDataPrefs=1, birth=1, merge=1)
  #for key in ArgDict:
  #  if key.count('Name') > 0:
  #    RelevantOpts[ ArgDict[key] ] = 1
  #print [k for k in ArgDict.keys()]
  for key in ArgDict:
    if key.count('Name') > 0:
      continue
    with open( os.path.join(taskoutpath, 'args-'+key+'.txt'), 'w') as fout:
      json.dump(ArgDict[key], fout)

########################################################### Config Subroutines
###########################################################
def createUniqueRandomSeed( jobname, taskID=0):
  ''' Get unique seed for a random number generator,
       deterministically using the jobname and taskID.
      Seed is reproducible on any machine, regardless of OS or 32/64 arch.
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
  
  
def getOutputPath(ReqArgs, KwArgs, taskID=0 ):
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
                       KwArgs['OutputPrefs']['jobname'], 
                       str(taskID) )

def createEmptyOutputPathOnDisk( taskoutpath ):
  ''' Create specified path (and all parent paths) on the file system,
      and make sure that path is empty (to avoid confusion with saves from previous runs).
  '''
  from distutils.dir_util import mkpath
  # Ensure the path (including all parent paths) exists
  mkpath( taskoutpath )
  # Ensure the path has no data from previous runs
  deleteAllFilesFromDir( taskoutpath )
  
def deleteAllFilesFromDir( savefolder, prefix=None ):
  '''  Erase (recursively) all contents of a folder
         so that we can write fresh results to it
  '''
  for the_file in os.listdir( savefolder ):
    if prefix is not None:
      if not the_file.startswith(prefix):
        continue
    file_path = os.path.join( savefolder, the_file)
    if os.path.isfile(file_path) or os.path.islink(file_path):
      os.unlink(file_path)

def configLoggingToConsoleAndFile(taskoutpath, doSaveToDisk=True, doWriteStdOut=True):
  Log.handlers = [] # remove pre-existing handlers!
  formatter = logging.Formatter('%(message)s')
  ###### Config logger to save transcript of log messages to plain-text file  
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
  ##### Config null logger, avoids error messages about no handler existing
  if not doSaveToDisk and not doWriteStdOut:
    Log.addHandler(logging.NullHandler())


if __name__ == '__main__':
  run()
