'''
HModel.py

Generic class for representing hierarchical Bayesian models in bnpy.

Attributes
-------
allocModel : a bnpy.allocmodel.AllocModel subclass
              such as MixModel, DPMixModel, etc.
             model for generating latent structure (such as cluster assignments) 
              
obsModel : a bnpy.obsmodel.ObsCompSet subclass
              such as GaussObsCompSet or ZMGaussObsCompSet
            model for generating observed data given latent structure

Key functions
-------
* calc_local_params
* get_global_suff_stats
* update_global_params
     
'''
from collections import defaultdict
import numpy as np
import os
import copy

import init
from allocmodel import AllocModelConstructorsByName
from obsmodel import ObsModelConstructorsByName
                   
class HModel(object):

  ######################################################### Constructors
  #########################################################    
  def __init__(self, allocModel, obsModel):
    ''' Constructor assembles HModel given fully valid subcomponents
    '''
    self.allocModel = allocModel
    self.obsModel = obsModel
    self.inferType = allocModel.inferType

  @classmethod
  def CreateEntireModel(cls, inferType, allocModelName, obsModelName, allocPriorDict, obsPriorDict, Data):
    ''' Constructor assembles HModel and all its submodels in one call
    '''
    AllocConstr = AllocModelConstructorsByName[allocModelName]
    allocModel = AllocConstr(inferType, allocPriorDict)

    ObsConstr = ObsModelConstructorsByName[obsModelName]
    obsModel = ObsConstr.CreateWithPrior(inferType, obsPriorDict, Data)
    return cls(allocModel, obsModel)
  
    
  def copy(self):
    ''' Create a clone of this object with distinct memory allocation
        Any manipulation of clone's parameters will NOT affect self
    '''
    return copy.deepcopy( self )

  ######################################################### Local Params
  #########################################################    
  def calc_local_params(self, Data, LP=None, **kwargs):
    ''' Calculate the local parameters specific to each data item,
          given global parameters.
        This is the E-step of the EM algorithm.        
    '''
    if LP is None:
      LP = dict()
    # Calculate the "soft evidence" each obsModel component has on each item
    # Fills in LP['E_log_soft_ev']
    LP = self.obsModel.calc_local_params(Data, LP, **kwargs)
    # Combine with allocModel probs of each cluster
    # Fills in LP['resp'], a Data.nObs x K matrix whose rows sum to one
    LP = self.allocModel.calc_local_params(Data, LP, **kwargs)
    return LP

  ######################################################### Suff Stats
  #########################################################   
  def get_global_suff_stats(self, Data, LP, doAmplify=False, **kwargs):
    ''' Calculate sufficient statistics for each component,
          given data and local responsibilities
        This is necessary prep for the M-step of EM algorithm.
    '''
    SS = self.allocModel.get_global_suff_stats(Data, LP, **kwargs)
    SS = self.obsModel.get_global_suff_stats(Data, SS, LP, **kwargs)
    if doAmplify:
      # Change effective scale of the suff stats, for soVB learning
      if hasattr(Data,"nDoc"):
        ampF = Data.nDocTotal / Data.nDoc
        SS.applyAmpFactor(ampF)
      else:
        ampF = Data.nObsTotal / Data.nObs
        SS.applyAmpFactor(ampF)

    return SS
  
  ######################################################### Global Params
  #########################################################   
  def update_global_params(self, SS, rho=None, **kwargs):
    ''' Update (in-place) global parameters given provided suff stats.
        This is the M-step of EM.
    '''
    self.allocModel.update_global_params(SS, rho, **kwargs)
    self.obsModel.update_global_params(SS, rho, **kwargs)
  
  def set_global_params(self, **kwargs):
    self.allocModel.set_global_params(**kwargs)
    self.obsModel.set_global_params(**kwargs)

  def insert_global_params(self, **kwargs):
    self.allocModel.insert_global_params(**kwargs)
    self.obsModel.insert_global_params(**kwargs)

  ######################################################### Evidence
  #########################################################     
  def calc_evidence(self, Data=None, SS=None, LP=None):
    ''' Compute the evidence lower bound (ELBO) of the objective function.
    '''
    if Data is not None and LP is None and SS is None:
      LP = self.calc_local_params(Data)
      SS = self.get_global_suff_stats(Data, LP)
    evA = self.allocModel.calc_evidence(Data, SS, LP)
    evObs = self.obsModel.calc_evidence(Data, SS, LP)
    return evA + evObs
  
  ######################################################### Init Global Params
  #########################################################    
  def init_global_params(self, Data, **initArgs):
    ''' Initialize (in-place) global parameters
    '''
    initname = initArgs['initname']
    if initname.count('true') > 0:
      init.FromTruth.init_global_params(self, Data, **initArgs)
    elif initname.count(os.path.sep) > 0:
      init.FromSaved.init_global_params(self, Data, **initArgs)
    elif str(type(self.obsModel)).count('Gauss') > 0:
      init.FromScratchGauss.init_global_params(self, Data, **initArgs)
    elif str(type(self.obsModel)).count('Mult') > 0:
      init.FromScratchMult.init_global_params(self, Data, **initArgs)
    elif str(type(self.obsModel)).count('BernRel') > 0:
      init.FromScratchBernRel.init_global_params(self, Data, **initArgs)
    else:
      raise NotImplementedError("TODO")

  ######################################################### I/O Utils
  ######################################################### 
  def getAllocModelName(self):
    return self.allocModel.__class__.__name__

  def getObsModelName(self):
    return self.obsModel.__class__.__name__

  def get_model_info( self ):
    s =  'Allocation Model:  %s\n'  % (self.allocModel.get_info_string())
    s += 'Obs. Data  Model:  %s\n' % (self.obsModel.get_info_string())
    s += 'Obs. Data  Prior:  %s' % (self.obsModel.get_info_string_prior())
    return s
