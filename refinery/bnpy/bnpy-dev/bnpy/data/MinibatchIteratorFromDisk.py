'''
MinibatchIteratorFromDisk.py

Extension of MinibatchIterator 
   reads in data that has been pre-split into batches on disk.
   
Usage:
  construct by providing a list of valid filepaths to .mat files (see XData for format)
  then call has_next_batch()
            get_next_batch()
  
  Traversal order of the files is randomized every lap through the full dataset
  Set the "dataseed" parameter to get repeatable orders.
'''

import numpy as np
import scipy.io

from MinibatchIterator import MinibatchIterator
from XData import XData

class MinibatchIteratorFromDisk( MinibatchIterator):

  def __init__(self):
    raise NotImplementedError("TODO")