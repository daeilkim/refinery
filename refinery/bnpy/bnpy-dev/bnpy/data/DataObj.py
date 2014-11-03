'''
DataObj.py

General abstract base class for all data objects,
whether they are full datasets or iterators over small batches of data

Attributes
-------
nObs : 
'''

class DataObj(object):
  @classmethod
  def read_from_mat(self, matfilepath):
    ''' Constructor for building data object from disk
    '''
    pass
  
  def __init__(self, *args, **kwargs):
    ''' Constructor for building data object from scratch in memory
    '''
    pass
    
  def get_short_name(self):
    ''' Returns string with short name (at most 10 char) of this data object,
          with no spaces and only alpha-numeric characters.
        Useful for creating filepaths specific for this data object.
    '''
    if hasattr(self, 'shortname'):
      return self.shortname
    return "MyData%d" % (self.nObs)

  def get_text_summary(self, **kwargs):
    ''' Returns string with human-readable description of this dataset 
        e.g. source, author/creator, etc.
    '''
    if hasattr(self, 'summary'):
      return self.summary
    s = '%s. nObs %d' % (self.__class__.__name__, self.nObs)
    return s

  def summarize_num_observations(self):
    ''' Returns string summary of number of observations in this data object
    '''
    pass
    
  def select_subset_by_mask(self, *args, **kwargs):
    ''' Returns DataObj of the same type, containing a subset of self's data
    '''
    pass
    
  def add_data(self, DataObj):
    ''' Updates (in-place) the dataset to include provided data
    '''
    pass