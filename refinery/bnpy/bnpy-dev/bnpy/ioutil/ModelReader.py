'''
ModelReader.py

Read in a bnpy model from disk

Related
-------
ModelWriter.py
'''
import numpy as np
import scipy.io
import os

from ModelWriter import makePrefixForLap
from bnpy.allocmodel import *
from bnpy.obsmodel import *
from bnpy.distr import *

GDict = globals()

def getPrefixForLapQuery(taskpath, lapQuery):
  ''' Search among the saved lap params in taskpath for the lap nearest query.

      Returns
      --------
      prefix : string like 'Lap0001.000' that indicates lap for saved parameters.
  '''
  saveLaps = np.loadtxt(os.path.join(taskpath,'laps-saved-params.txt'))
  distances = np.abs(lapQuery - saveLaps)
  bestLap = saveLaps[np.argmin(distances)]
  return makePrefixForLap(bestLap), bestLap

def loadModelForLap(matfilepath, lapQuery):
  ''' Loads saved model with lap closest to provided lapQuery
      Returns
      -------
      model, true-lap-id
  '''
  prefix, bestLap = getPrefixForLapQuery(matfilepath, lapQuery)
  model = load_model(matfilepath, prefix=prefix)
  return model, bestLap

def load_model( matfilepath, prefix='Best'):
  ''' Load model stored to disk by ModelWriter
  '''
  # avoids circular import
  import bnpy.HModel as HModel
  obsModel = load_obs_model(matfilepath, prefix)
  allocModel = load_alloc_model(matfilepath, prefix)
  return HModel(allocModel, obsModel)
  
def load_alloc_model(matfilepath, prefix):
  apriorpath = os.path.join(matfilepath,'AllocPrior.mat')
  amodelpath = os.path.join(matfilepath,prefix+'AllocModel.mat')
  APDict = loadDictFromMatfile(apriorpath)
  ADict = loadDictFromMatfile(amodelpath)
  AllocConstr = GDict[ADict['name']]
  amodel = AllocConstr( ADict['inferType'], APDict )
  amodel.from_dict( ADict)
  return amodel
  
def load_obs_model(matfilepath, prefix):
  obspriormatfile = os.path.join(matfilepath,'ObsPrior.mat')
  PDict = loadDictFromMatfile(obspriormatfile)
  if PDict['name'] == 'NoneType':
    obsPrior = None
  else:
    PriorConstr = GDict[PDict['name']]
    obsPrior = PriorConstr( **PDict)
  obsmodelpath = os.path.join(matfilepath,prefix+'ObsModel.mat')
  ODict = loadDictFromMatfile(obsmodelpath)

  ObsConstr = GDict[ODict['name']]
  CompDicts = get_list_of_comp_dicts( ODict['K'], ODict)
  return ObsConstr.CreateWithAllComps( ODict, obsPrior, CompDicts)
  
def get_list_of_comp_dicts( K, Dict ):
  ''' We store all component params stacked together in an array.
      This function extracts them into individual components.
  '''
  MyList = [ dict() for k in xrange(K)]
  for k in xrange(K):
    for key in Dict:
      if type(Dict[key]) is not np.ndarray:
        continue
      x = Dict[key]
      if K == 1 and (key != 'min_covar' and key != 'K'):
        MyList[k][key] = x.copy()
      elif x.ndim == 1 and x.size > 1:
        MyList[k][key] = x[k].copy()
      elif x.ndim == 2:
        MyList[k][key] = x[:,k].copy()
      elif x.ndim == 3:
        MyList[k][key] = x[:,:,k].copy()
  return MyList
  
def loadDictFromMatfile(matfilepath):
  ''' Returns
      --------
       dict D where all numpy entries have good byte order, flags, etc.
  '''
  Dtmp = scipy.io.loadmat( matfilepath )
  D = dict( [x for x in Dtmp.items() if not x[0].startswith('__')] )
  for key in D:
    if type( D[key] ) is not np.ndarray:
      continue
    x = D[key]
    if x.ndim == 1:
      x = x[0]
    elif x.ndim == 2:
      x = np.squeeze(x)
    D[key] = x.newbyteorder('=').copy()
  return D
