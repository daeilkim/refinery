'''
Generic unit tests for end-to-end model training with bnpy.

----------------- EM
* verify evidence monotonically increasing, repeatable with same seed
* verify run started at ideal params ("from truth") does not wander away
* verify runs 'from scratch' finds ideal params, for some fraction of all runs

----------------- VB
* verify evidence monotonically increasing, repeatable with same seed
* verify run started at ideal params ("from truth") does not wander away
TODO
* verify runs 'from scratch' finds ideal params, for some fraction of all runs
TODO

'''
import sys
import numpy as np
import unittest
from unittest.case import SkipTest

import bnpy.viz 
from bnpy.util import closeAtMSigFigs
from bnpy.Run import run
from bnpy.ContinueRun import continueRun
import Util

class AbstractEndToEndTest(unittest.TestCase):
  __test__ = False # Do not execute this abstract module!
  def shortDescription(self):
    return None

  ######################################################### EM tests
  #########################################################
  def test_EM__evidence_repeatable_and_monotonic(self):
    if 'EM' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'EM', **kwargs)
    hmodel2, LP2, Info2 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'EM', **kwargs)
    assert len(Info1['evTrace']) == len(Info2['evTrace'])
    assert np.allclose( Info1['evTrace'], Info2['evTrace'])

    assert self.verify_monotonic(Info1['evTrace'])

  def test_EM__fromTruth(self):
    ''' Verify EM alg will not drastically alter model init'd to ideal params.

        Performs one run of EM on provided dataset, using 'trueparams' init.
    '''
    print '' # new line for nosetests
    if 'EM' not in self.learnAlgs or not hasattr(self, 'TrueParams'):
      raise SkipTest
    # Create keyword args for "trueparams" initialization
    kwargs = self.get_kwargs(initname='trueparams')
    self.Data.TrueParams = self.TrueParams
    self.Data.TrueParams['K'] = self.K
    # Run EM from a "trueparams" init
    model, LP, Info = run(self.Data, self.allocModelName, self.obsModelName,
                          'EM', **kwargs)
    assert self.verify_monotonic(Info['evTrace'])
    for key in self.TrueParams:
      if key == 'K': 
        continue
      elif hasattr(model.allocModel, key):
        print '--------------- %s' % (key)
        arrEst = getattr(model.allocModel, key)
        arrTrue = self.TrueParams[key]
        Util.pprint(arrTrue, 'true')
        Util.pprint(arrEst, 'est')
        assert self.verify_close(arrTrue, arrEst, key)
      
      elif hasattr(model.obsModel.comp[0], key):
        for k in range(self.K):

          arrTrue = self.TrueParams[key][k]
          arrEst = getattr(model.obsModel.comp[k], key)
          print '--------------- %d/%d %s' % (k+1, self.K, key)
          Util.pprint(arrTrue, 'true')
          Util.pprint(arrEst, 'est')
          assert self.verify_close(arrTrue, arrEst, key)

  def test_EM__fromScratch(self):
    ''' Verify EM alg can estimate params near 'ideal', over many runs.

        Performs fromScratchTrials trials, and verifies some fraction succeed.
        Local optima issues will prevent all runs from reaching the ideal.
    '''
    if 'EM' not in self.learnAlgs or not hasattr(self, 'TrueParams'):
      raise SkipTest
    if not hasattr(self, 'fromScratchTrials'):
      self.fromScratchTrials = 5
      self.fromScratchSuccessRate = 0.4
    successMask = np.zeros(self.fromScratchTrials)
    # Run many trials of EM, record each as 0/1 (failure/success)
    for task in range(self.fromScratchTrials):
      successMask[task] = self.run_EM__fromScratch(task+1)
    nSuccess_expected = self.fromScratchSuccessRate * self.fromScratchTrials
    nSuccess_observed = successMask.sum()
    assert nSuccess_observed >= nSuccess_expected
      
  def run_EM__fromScratch(self, taskid=0):
    ''' Returns True/False for whether single run of EM finds ideal params.
    '''
    print '' # new line for nosetests
    print '============================================== task %d' % (taskid)
    kwargs = self.get_kwargs()
    kwargs.update(self.fromScratchArgs)
    model, LP, Info = run(self.Data, self.allocModelName, self.obsModelName,
                          'EM', taskid=taskid, **kwargs)
    assert self.verify_monotonic(Info['evTrace'])

    # First, find perm of estimated comps to true comps
    #  if one exists, by using obsmodel params
    permIDs = None
    allocKeyList = []
    for key in self.TrueParams:
      if key == 'K': 
        continue
      elif hasattr(model.obsModel.comp[0], key):
        arrTrue = self.TrueParams[key]
        arrEst = Util.buildArrForObsModelParams(model.obsModel.comp, key)
        if permIDs is None:
          isG, permIDs = self.verify_close_under_some_perm(arrTrue, arrEst, key)
          if not isG:
            print ' FAILED TO FIND IDEAL PARAMS'
            argstring = ' '.join(sys.argv[1:])
            if 'nocapture' in argstring:
              from matplotlib import pylab
              bnpy.viz.GaussViz.plotGauss2DFromHModel(model)
              pylab.show()
            return False
        arrEst = arrEst[permIDs]
        assert self.verify_close(arrTrue, arrEst, key)  
      else:
        allocKeyList.append(key)

    assert permIDs is not None
    for key in allocKeyList:
      print '--------------- %s' % (key)
      arrTrue = self.TrueParams[key]
      arrEst = getattr(model.allocModel, key)[permIDs]
      Util.pprint(arrTrue, 'true')
      Util.pprint(arrEst, 'est')
      assert self.verify_close(arrTrue, arrEst, key)
    return True

  ######################################################### VB tests
  #########################################################
  def test_vb_repeatable_and_monotonic(self):
    ''' Verify VB runs with same seed produce exact same output, monotonic ELBO.
    '''
    if 'VB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    hmodel2, LP2, Info2 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    assert len(Info1['evTrace']) == len(Info2['evTrace'])
    assert np.allclose( Info1['evTrace'], Info2['evTrace'])
    assert self.verify_monotonic(Info1['evTrace'])

  def test_vb_repeatable_when_continued(self):
    if 'VB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs_do_save_disk()
    kwargs['nLap'] = 10
    hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    kwargs['nLap'] = 5
    kwargs['startLap'] = 5
    hmodel2, LP2, Info2 = continueRun(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    if hasattr(self, 'mustRetainLPAcrossLapsForGuarantees'):
      print Info1['evTrace'][-1]
      print Info2['evTrace'][-1]
      assert closeAtMSigFigs(Info1['evTrace'][-1], Info2['evTrace'][-1], M=2)

    else:
      assert Info1['evTrace'][-1] == Info2['evTrace'][-1]

  ######################################################### soVB tests
  #########################################################

  def test_sovb_repeatable_across_diff_num_batches(self):
    if 'soVB' not in self.learnAlgs:
      raise SkipTest
    for nBatch in [1, 2, 3]:
      kwargs = self.get_kwargs()
      kwargs['nBatch'] = nBatch
      hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
      hmodel2, LP2, Info2 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
      assert len(Info1['evTrace']) == len(Info2['evTrace'])
      assert np.allclose( Info1['evTrace'], Info2['evTrace'])

  def test_sovb_with_one_batch_equivalent_to_vb(self):
    if 'soVB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 1
    kwargs['rhoexp'] = 0
    __, __, vbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, sovbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
    vbEv = vbInfo['evTrace'][:-1]
    sovbEv = sovbInfo['evTrace']
    for ii in range(len(vbEv)):
      if hasattr(self, 'mustRetainLPAcrossLapsForGuarantees'):
        print vbEv[ii], sovbEv[ii]
        assert closeAtMSigFigs(vbEv[ii], sovbEv[ii], M=2)
      else:
        assert closeAtMSigFigs(vbEv[ii], sovbEv[ii], M=8)

  ######################################################### moVB tests
  #########################################################

  def test_movb_repeatable_across_diff_num_batches(self):
    if 'moVB' not in self.learnAlgs:
      raise SkipTest
    for nBatch in [1, 2, 3]:
      kwargs = self.get_kwargs()
      kwargs['nBatch'] = nBatch
      hmodel1, LP1, Info1 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
      hmodel2, LP2, Info2 = run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
      assert len(Info1['evTrace']) == len(Info2['evTrace'])
      assert np.allclose( Info1['evTrace'], Info2['evTrace'])

  def test_movb_with_one_batch_equivalent_to_vb(self):
    if 'moVB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 1
    __, __, vbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, movbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
    vbEv = vbInfo['evTrace'][:-1]
    movbEv = movbInfo['evTrace']
    print vbEv
    print movbEv
    assert len(vbEv) == len(movbEv)
    for ii in range(len(vbEv)):
      assert closeAtMSigFigs(vbEv[ii], movbEv[ii], M=8)

  ######################################################### ELBO tests
  #########################################################

  def test_vb_sovb_and_movb_all_estimate_evBound_in_same_ballpark(self):
    if 'moVB' not in self.learnAlgs:
      raise SkipTest
    kwargs = self.get_kwargs()
    kwargs['nBatch'] = 5
    __, __, vbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'VB', **kwargs)
    __, __, movbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'moVB', **kwargs)
    __, __, sovbInfo = run(self.Data, self.allocModelName,
                          self.obsModelName, 'soVB', **kwargs)
    vbEv = vbInfo['evBound']
    movbEv = movbInfo['evBound']
    sovbEv = np.mean(sovbInfo['evTrace'][-10:])

    print vbEv
    print movbEv
    print sovbEv
    assert closeAtMSigFigs(vbEv, movbEv, M=2)
    assert closeAtMSigFigs(vbEv, sovbEv, M=2)

  ######################################################### keyword accessors
  #########################################################
  def get_kwargs(self, **kwargsIN):
    ''' Keyword args for calling bnpy.run, avoiding saving to disk
    '''
    kwargs = dict(saveEvery=-1, printEvery=-1, traceEvery=1)
    kwargs['doSaveToDisk'] = False
    kwargs['doWriteStdOut'] = False
    kwargs['convergeSigFig'] = 12
    if hasattr(self, 'kwargs'):
      kwargs.update(self.kwargs)
    kwargs.update(**kwargsIN)
    return kwargs

  def get_kwargs_do_save_disk(self):
    ''' Keyword arguments for bnpy.run when tests need to write to disk
    '''
    kwargs = dict(saveEvery=1, printEvery=1, traceEvery=1)
    kwargs['doSaveToDisk'] = True
    kwargs['doWriteStdOut'] = False
    if hasattr(self, 'kwargs'):
      kwargs.update(self.kwargs)
    return kwargs


  ######################################################### verify_close
  #########################################################
  def verify_close(self, arrTrue, arrEst, key):
    ''' Returns True if two provided arrays are close, False otherwise.
    '''
    arrTrue = arrTrue.copy()
    arrEst = arrEst.copy()
    if hasattr(self, 'ProxFunc') and key in self.ProxFunc:
      mask = self.ProxFunc[key](arrTrue, arrEst)
    else:
      mask = np.allclose(arrTrue, arrEst)
    if np.all(mask):
      return True
    return False

  def verify_close_under_some_perm(self, arrTrue, arrEst, key, **kwargs):
    ''' Returns True if arrays are numerically close under some permutation.
    '''
    arrTrue = np.asarray(arrTrue).copy()
    arrEst = np.asarray(arrEst).copy()
    K = arrEst.shape[0]
    K2 = arrTrue.shape[0]
    assert K == K2
    true2est = -1 * np.ones(K, dtype=np.int32)
    est2true = -1 * np.ones(K, dtype=np.int32)
    for k in range(K):
      for c in range(K):
        if est2true[c] >= 0:
          continue
        if self.verify_close(arrTrue[k], arrEst[c], key, **kwargs):
          true2est[k] = c
          est2true[c] = k
          break
    return np.all(true2est >= 0), true2est

  def pprint_mismatched_entries(self, arrTrue, arrEst, key, replaceVal=-123):
    arrTrue = arrTrue.copy()
    arrEst = arrEst.copy()
    if key in self.ProxFunc:
      mask = self.ProxFunc[key](arrTrue, arrEst)
    else:
      mask = np.allclose(arrTrue, arrEst)
    arrTrue[mask] = replaceVal
    arrEst[mask] = replaceVal
    Util.pprint(arrTrue, 'true', replaceVal=replaceVal)
    Util.pprint(arrEst, 'est', replaceVal=replaceVal)

  def test__verify_close_under_some_perm(self):
    isG, permIDs = self.verify_close_under_some_perm([1,2,3,4], [4,3,2,1], '')
    assert isG
    assert np.allclose(permIDs, [3,2,1,0])

    avec = np.asarray([1,2,3,4,5,6,7,8])
    bvec = np.asarray([2,4,6,8,1,3,5,7])
    isG, permIDs = self.verify_close_under_some_perm(avec, bvec, '')
    assert isG
    print permIDs
    assert np.allclose(permIDs, [4, 0, 5, 1, 6, 2, 7, 3])
    assert np.allclose(avec, bvec[permIDs])

    avec = np.asarray([1,2,3,4])
    bvec = np.asarray([2,4,6,8])
    isG, permIDs = self.verify_close_under_some_perm(avec, bvec, '')
    assert not isG

  ######################################################### verify_monotonic
  #########################################################
  def verify_monotonic(self, ELBOvec):
    ''' Returns True if monotonically increasing, False otherwise.
    '''
    ELBOvec = np.asarray(ELBOvec, dtype=np.float64)
    assert ELBOvec.ndim == 1
    diff = ELBOvec[1:] - ELBOvec[:-1]
    maskIncrease = diff > 0
    maskWithinPercDiff = np.abs(diff)/np.abs(ELBOvec[:-1]) < 0.0000001
    mask = np.logical_or(maskIncrease, maskWithinPercDiff)
    mask = np.asarray(mask, dtype=np.float64)
    return np.abs(np.sum(mask) - float(diff.size)) < 0.000001

  def test__verify_monotonic_catches_bad(self):
    assert self.verify_monotonic( [502.3, 503.1, 504.01, 504.00999999])
    assert not self.verify_monotonic( [502.3, 503.1, 504.01, 504.00989999])
    assert not self.verify_monotonic( [401.3, 400.99, 405.12])
