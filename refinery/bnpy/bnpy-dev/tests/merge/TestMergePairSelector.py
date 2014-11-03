'''
Unit tests for MergePairSelector.py

Verifies that we can successfully select components to merge
'''
import numpy as np
import unittest

from bnpy.learnalg import MergeTracker
from bnpy.learnalg import MergePairSelector

class TestMergePairSelector(unittest.TestCase):
  def shortDescription(self):
    return None

  def setUp(self):
    pass

  def test_reindexAfterMerge(self):
    MSelector = MergePairSelector()
    MSelector.MScores[0] = 5
    MSelector.MScores[3] = 5
    MSelector.MScores[4] = 5
    MSelector.PairMScores[(0,1)] = 5
    MSelector.PairMScores[(3,4)] = 5
    MSelector.PairMScores[(5,6)] = 5

    MSelector.reindexAfterMerge(2,3)

    assert MSelector.MScores[0] == 5
    assert MSelector.MScores[3] == 5
    assert 2 not in MSelector.MScores

    assert len(MSelector.PairMScores.keys()) == 2
    assert (0,1) in MSelector.PairMScores
    assert (4,5) in MSelector.PairMScores
    assert (2,3) not in MSelector.PairMScores

  def test_select_merge_components_random(self):
    ''' Verify that under random choices, we select among 3 components
          equally often
    '''
    MT = MergeTracker(3)
    MSelector = MergePairSelector()
    counts = np.zeros(3)
    for trial in range(1000):
      kA, kB = MSelector.select_merge_components(None, None, MT, mergename='random')
      counts[kA] += 1
      counts[kB] += 1
    counts /= np.sum(counts)
    minFrac = 0.25
    maxFrac = 0.4
    # Uniform at random means fraction of choice should be ~1/3 for each
    assert np.all(counts > minFrac)
    assert np.all(counts < maxFrac)

  def test_select_merge_components_random_raisesError(self):
    ''' Verify that when comp 0 is excluded with K=3
          we cannot provide comp 0 as kA, [error is raised]
          AND
          in free choice, we only choose kA=1, kB=2
    '''
    MT = MergeTracker(3)
    MSelector = MergePairSelector()

    MT.excludeList = set([0])
    MT._synchronize_and_verify()
    for trial in range(10):
      kA, kB = MSelector.select_merge_components(None, None, MT, kA=1, mergename='random')
      assert kA == 1
      assert kB == 2
    for trial in range(10):
      kA, kB = MSelector.select_merge_components(None, None, MT, kA=2, mergename='random')
      assert kA == 1
      assert kB == 2
    with self.assertRaises(AssertionError):
      kA, kB = MSelector.select_merge_components(None, None, MT, mergename='random', kA=0)
    
  def test_select_merge_components_random_raisesErrorAllButOneExcluded(self):
    ''' Verify that when comps 0,1 are excluded with K=3
          we cannot provide comp 2 as kA, [error is raised]
    '''
    MT = MergeTracker(3)
    MSelector = MergePairSelector()

    MT.excludeList = set([1, 0])
    MT._synchronize_and_verify()
    with self.assertRaises(AssertionError):
      kA, kB = MSelector.select_merge_components(None, None, MT, mergename='random', kA=2)
    