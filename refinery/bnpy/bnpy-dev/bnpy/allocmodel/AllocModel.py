''' AllocModel.py
'''
from __future__ import division

class AllocModel(object):

  ######################################################### Constructors
  #########################################################
  def __init__(self, inferType):
    self.inferType = inferType

  def set_prior(self, **kwargs):
    pass  
    
  ######################################################### Accessors
  #########################################################
  def get_keys_for_memoized_local_params(self):
    ''' Return list of string names of the LP fields
        that this object needs to memoize across visits to a particular batch
    '''
    return list()

  def requireMergeTerms(self):
    ''' Return boolean indicator for whether this model
         requires precomputed merge terms
    '''
    return True

  ######################################################### Local Params
  #########################################################
  def calc_local_params( self, Data, LP ):
    ''' 
    '''
    pass

  ######################################################### Suff Stats
  #########################################################
  def get_global_suff_stats( self, Data, SS, LP ):
    ''' 
    '''
    pass

  ######################################################### Global Params
  #########################################################
  def update_global_params( self, SS, rho=None, **kwargs ):
    ''' Update (in-place) global parameters for this allocation model object,
        given the provided suff stats object SS
        This is the M-step of EM/VB algorithm
    '''
    self.K = SS.K
    if self.inferType == 'EM':
      self.update_global_params_EM(SS)
    elif self.inferType == 'VB' or self.inferType == "moVB":
      self.update_global_params_VB(SS, **kwargs)
    elif self.inferType == 'soVB':
      if rho is None or rho==1:
        self.update_global_params_VB(SS, **kwargs)
      else: 
        self.update_global_params_soVB(SS, rho, **kwargs)
    else:
      raise ValueError( 'Unrecognized Inference Type! %s' % (self.inferType) )
 
 
  ######################################################### Evidence
  #########################################################
  def calc_evidence(self):
    pass

  ######################################################### IO Utils
  #########################################################   for humans
  def get_info_string( self):
    ''' Returns one-line human-readable terse description of this object
    '''
    pass

  ######################################################### IO Utils
  #########################################################   for machines
  def to_dict_essential(self):
    PDict = dict(name=self.__class__.__name__, inferType=self.inferType)
    if hasattr(self,'K'):
      PDict['K'] = self.K
    return PDict
    
  def to_dict(self):
    pass
  
  def from_dict(self):
    pass
 
  def get_prior_dict(self):
    pass

