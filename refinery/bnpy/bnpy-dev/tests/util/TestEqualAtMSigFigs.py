'''
Unit tests for closeAtMSigFigs
'''

EPS= 1e-10

from bnpy.util import closeAtMSigFigs
import numpy as np
class TestCloseAtMSigFigs(object):

  def test_close_at_2_sig_figs_boundary(self, M=2):
    assert closeAtMSigFigs( 9.96, 9.98, M=M, tol=5)
    # once one arg goes about 10, the new threshold is 0.5, not 0.05
    assert closeAtMSigFigs( 9.96, 10.01, M=M, tol=5)
    assert closeAtMSigFigs( 9.96, 10.46, M=M, tol=5)
    assert not closeAtMSigFigs( 9.96, 10.4601, M=M, tol=5)

  def test_close_at_2_sig_figs_readable(self, M=2):
    assert closeAtMSigFigs( 1.42, 1.42, M=M, tol=5)
    assert closeAtMSigFigs( 1.42, 1.47, M=M, tol=5)
    assert not closeAtMSigFigs( 1.42, 1.4701, M=M, tol=5)
    assert closeAtMSigFigs( 1.42, 1.37, M=M, tol=5)
    assert not closeAtMSigFigs( 1.42, 1.36999, M=M, tol=5)

    assert closeAtMSigFigs( 1.42e5, 1.47e5, M=M, tol=5)
    assert not closeAtMSigFigs( 1.42e5, 1.4701e5, M=M, tol=5)

    assert closeAtMSigFigs( 1.42e5, 1.37e5, M=M, tol=5)
    assert not closeAtMSigFigs( 1.42e5, 1.3699e5, M=M, tol=5)

    assert closeAtMSigFigs( -1.42e2, -1.47e2, M=M, tol=5)
    assert not closeAtMSigFigs( -1.42e2, -1.4701e2, M=M, tol=5)

    assert closeAtMSigFigs( -1.42e2, -1.37e2, M=M, tol=5)
    assert not closeAtMSigFigs( -1.42e2, -1.369999e2, M=M, tol=5)


  def test_close_at_2_sig_figs_exhaustive(self, M=2):
    for a in np.arange(-9., 9., 0.01):
      for offset in np.arange( 0.05000001, 0.1, 0.01):
        isEqual = closeAtMSigFigs(a, a+offset, M=M, tol=5)
        assert not isEqual

      for offset in np.arange( -0.05000001, -0.1, -0.01):
        isEqual = closeAtMSigFigs(a, a+offset, M=M, tol=5)
        assert not isEqual

      for offset in np.arange( -0.05, 0.05+EPS, 0.01):
        isEqual = closeAtMSigFigs(a, a+offset, M=M, tol=5)
        if not isEqual:
          print a
          print a+offset
        assert isEqual



  def test_close_at_3_sig_figs_exhaustive(self, M=3):
    for a in np.arange(-9., 9., 0.01):
      for offset in np.arange( 0.005000001, 0.01, 0.001):
        isEqual = closeAtMSigFigs(a, a+offset, M=M, tol=5)
        assert not isEqual

      for offset in np.arange( -0.005000001, -0.01, -0.001):
        isEqual = closeAtMSigFigs(a, a+offset, M=M, tol=5)
        assert not isEqual

      for offset in np.arange( -0.005, 0.005+EPS, 0.001):
        isEqual = closeAtMSigFigs(a, a+offset, M=M, tol=5)
        if not isEqual:
          print a
          print a+offset
        assert isEqual


  def test_close_at_6_sig_figs_exhaustive(self, M=6):
    for a in np.arange(-9., 9., 0.01):
      for offset in np.arange( 5.000001e-6, 2*5.000001e-6, 1e-6):
        isEqual = closeAtMSigFigs(a, a+offset, M=M, tol=5)
        assert not isEqual

      for offset in np.arange( -5e-6, 5e-6+EPS, 1e-6):
        isEqual = closeAtMSigFigs(a, a+offset, M=M, tol=5)
        if not isEqual:
          print a
          print a+offset
        assert isEqual