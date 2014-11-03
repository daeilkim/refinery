'''
Unit tests for MergeTracker.py

Verification tracking of which comps have been merged already
 works as expected and produces valid models.


'''
import numpy as np
import unittest

from bnpy.learnalg import MergeTracker

class TestMergeTracker(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    pass

  def test_recordMergeResult_assertRaisesOnRepeatPair(self):
    MT = MergeTracker(4)
    print MT.excludeList
    MT.recordResult(0, 1, True)
    with self.assertRaises(AssertionError):
      MT.recordResult(0, 1, True)

  def test_recordMergeResult_assertRaisesWhenCompAlreadyPartOfMerge(self):
    MT = MergeTracker(4)
    print MT.excludeList
    MT.recordResult(2, 3, True)
    with self.assertRaises(AssertionError):
      MT.recordResult(0, 2, False)
    with self.assertRaises(AssertionError):
      MT.recordResult(1, 2, False)

  def test_recordMergeResult_assertRaisesOnRepeatPair2(self):
    MT = MergeTracker(6)
    MT.recordResult(0, 1, False)
    MT.recordResult(0, 2, False)
    MT.recordResult(0, 3, False)
    MT.recordResult(0, 4, True)
    MT.recordResult(1, 2, True)
    assert len(MT.excludePairs[1]) == MT.K
    with self.assertRaises(AssertionError):
      MT.recordResult(1, 2, False)

  def test_recordMergeResult(self):
    MT = MergeTracker(6)
    MT.recordResult(0, 1, False)
    MT.recordResult(0, 2, False)
    MT.recordResult(0, 3, False)
    assert len(MT.excludeList) == 0
    MT.recordResult(0, 4, True)
    assert 0 in MT.excludeList
    assert 1 not in MT.excludeList
    MT.recordResult(1, 2, True)
    assert 1 in MT.excludeList
    assert 2 not in MT.excludeList
    MT.recordResult(2, 3, True)
    assert 2 in MT.excludeList

    assert MT.K == 3
    assert MT.OrigK == 6
    assert (0,4) in MT.acceptedOrigIDs
    assert (1,2) in MT.acceptedOrigIDs
    assert (3,5) in MT.acceptedOrigIDs


  def test_synchronize_catch_former_bug1(self):
    ''' Given un-synched excludeList and excludePairs,
          verify that the synchronization will discover (correctly)
          that no pairs are left
        This prevents relapse of a bug captured in Jan 2013
    '''
    MT = MergeTracker(6)
    MT.excludeList = set([0, 2, 1, 4])
    MT.excludePairs[0] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[1] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[2] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[3] = set([0, 1, 2, 3, 5])
    MT.excludePairs[4] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[5] = set([0, 1, 2, 3, 5])
    MT._synchronize_and_verify()
    for k in range(6):
      assert k in MT.excludeList
    assert not MT.hasAvailablePairs()

  def test_synchronize_catch_former_bug2(self):
    ''' Given un-synched excludeList and excludePairs,
          verify that the synchronization will discover (correctly)
          that no pairs are left
        This prevents relapse of a bug captured in Jan 2013
    '''
    MT = MergeTracker(6)
    MT.excludeList = set([1, 4, 2, 3])
    MT.excludePairs[0] = set([0, 1, 3, 4, 5])
    MT.excludePairs[1] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[2] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[3] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[4] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[5] = set([0, 1, 3, 4, 5])
    MT._synchronize_and_verify()
    for k in range(6):
      assert k in MT.excludeList
    assert not MT.hasAvailablePairs()


  def test_synchronize_catch_former_bug3(self):
    ''' 
        This prevents relapse of a bug captured in Jan 2013
    '''
    MT = MergeTracker(7)
    MT.excludeList = set([3, 0, 2, 6])
    MT.excludePairs[0] = set([0, 1, 2, 3, 4, 5, 6])
    MT.excludePairs[1] = set([0, 1, 2, 3, 5])
    MT.excludePairs[2] = set([0, 1, 2, 3, 4, 5, 6])
    MT.excludePairs[3] = set([0, 1, 2, 3, 4, 5, 6])
    MT.excludePairs[4] = set([0, 2, 3, 4, 5])
    MT.excludePairs[5] = set([0, 1, 2, 3, 4, 5])
    MT.excludePairs[6] = set([0, 1, 2, 3, 4, 5, 6])
    MT._synchronize_and_verify()
    assert 1 in MT.getAvailableComps()
    assert 4 in MT.getAvailableComps()
    assert 5 in MT.excludePairs[1]
    assert 1 in MT.excludePairs[5]
    assert 6 in MT.excludePairs[4]
    assert 6 in MT.excludePairs[1]
