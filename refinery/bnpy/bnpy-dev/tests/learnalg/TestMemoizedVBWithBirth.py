'''
Unit tests for moVB with birth moves.

Coverage
--------
* do_birth_at_lap
  * verify births occur at the expected times (when lap < fracLapsBirth*nLap)
'''
import bnpy
import unittest

class TestMOVBWithBirth(unittest.TestCase):

  def setUp(self):
    birthP = dict(fracLapsBirth=0.8)
    algP = dict(nLap=10, birth=birthP)
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=dict())    

  def test_do_birth_at_lap(self):
    assert self.learnAlg.do_birth_at_lap(0)
    assert self.learnAlg.do_birth_at_lap(0.5)
    assert self.learnAlg.do_birth_at_lap(1)
    assert self.learnAlg.do_birth_at_lap(2)
    assert self.learnAlg.do_birth_at_lap(8)
    assert not self.learnAlg.do_birth_at_lap(8.05)
    assert not self.learnAlg.do_birth_at_lap(8.2)
    assert not self.learnAlg.do_birth_at_lap(9)
    assert not self.learnAlg.do_birth_at_lap(10)
    assert not self.learnAlg.do_birth_at_lap(11111)


class TestMOVBWithBirthFracThatNeedsRounding(TestMOVBWithBirth):
  ''' Now check it with a fraction that will need to be rounded.
  '''

  def setUp(self):
    birthP = dict(fracLapsBirth=0.7777)
    algP = dict(nLap=10, birth=birthP)
    self.learnAlg = bnpy.learnalg.MemoizedOnlineVBLearnAlg(savedir=None, seed=0, 
                              algParams=algP, outputParams=dict())    

