'''
Unit tests for MinibatchIterator.py
'''
import numpy as np
import unittest
import copy
import bnpy.data.XData as XData
import bnpy.data.MinibatchIterator as MinibatchIterator

class TestMinibatchIterator(unittest.TestCase):
  def shortDescription(self):
    return None
    
  def setUp(self):
    X = np.random.randn(100, 3)
    self.Data = XData(X=X)
    self.DataIterator = MinibatchIterator(self.Data, nBatch=10, nLap=10)
  
  def test_first_batch(self):
    assert self.DataIterator.has_next_batch()
    bData = self.DataIterator.get_next_batch()
    assert self.DataIterator.curLapPos == 0
    self.verify_batch(bData)
  
  def test_num_laps(self):
    ''' Make sure we raise the expected exception after exhausting all the data
    '''
    nLap = self.DataIterator.nLap
    nBatch = self.DataIterator.nBatch
    for lapID in range(nLap):
      for batchCount in range(nBatch):
        bData = self.DataIterator.get_next_batch()
        assert self.DataIterator.curLapPos == batchCount
        assert self.DataIterator.lapID == lapID
        self.verify_batch(bData)
    try:
      bData = self.DataIterator.get_next_batch()
      raise Exception('should not make it to this line!')
    except StopIteration:
      assert 1==1
        
  def test_batchIDs_traversal_order(self):
    ''' Make sure batchIDs from consecutive laps are not the same
    '''
    self.DataIterator.lapID = 0
    self.DataIterator.curLapPos = -1
    bData1 = self.DataIterator.get_next_batch()      
    batchOrder = copy.copy(self.DataIterator.batchOrderCurLap)
    
    self.DataIterator.lapID = 1
    self.DataIterator.curLapPos = -1
    bData2 = self.DataIterator.get_next_batch()      
    batchOrder2 = self.DataIterator.batchOrderCurLap
    print batchOrder, batchOrder2
    assert not np.allclose(batchOrder, batchOrder2)
    assert np.allclose(np.unique(batchOrder),np.unique(batchOrder2))
        
  
  def test_obs_full_coverage(self):
    ''' Make sure all data items are covered every lap
    '''
    coveredIDs = list()
    nBatch = self.DataIterator.nBatch
    for bID in range(nBatch):
      bData = self.DataIterator.get_next_batch()      
      obsIDs = self.DataIterator.getObsIDsForCurrentBatch()
      coveredIDs.extend(obsIDs)
    assert len(np.unique(coveredIDs)) == self.Data.nObsTotal
        
  def verify_batch(self, bData):
    assert bData.nObs == self.Data.nObs / self.DataIterator.nBatch
    assert bData.nObsTotal == self.Data.nObsTotal
    # Check that the data is as expected!
    batchX = bData.X    
    trueMask = self.DataIterator.getObsIDsForCurrentBatch()
    trueX = self.Data.X[trueMask]
    assert np.allclose(batchX, trueX)
