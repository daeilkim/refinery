'''
FromTruth.py

Initialize params of a bnpy model using "ground truth" information,
such as human annotations 

These are provided within a Data object, as a "TrueLabels" field
'''
import numpy as np
import FromScratchMult

def init_global_params(hmodel, Data, initname=None, seed=0, nRepeatTrue=2, **kwargs):
  ''' Initialize (in-place) the global params of the given hmodel
      using the true labels associated with the Data

      Args
      -------
      hmodel : bnpy model object to initialize
      Data   : bnpy Data object whose dimensions must match resulting hmodel
      initname : string name for the routine to use
                 'truelabels' or 'repeattruelabels'
  '''
  PRNG = np.random.RandomState(seed)
  if initname.count('truelabels') > 0:
    if hasattr(Data, 'TrueLabels'):
      resp = calc_resp_from_true_labels(Data)
    elif hasattr(Data, 'TrueParams'):
      if 'resp' in Data.TrueParams:
        resp = Data.TrueParams['resp']
      if 'word_variational' in Data.TrueParams:
        resp = Data.TrueParams['word_variational']
      

  if initname == 'truelabels':
    pass # have everything we need
  elif initname == 'repeattruelabels':
    Ktrue = resp.shape[1]
    rowIDs = PRNG.permutation(Data.nObs)
    L = len(rowIDs)/nRepeatTrue
    bigResp = np.zeros((Data.nObs, Ktrue*nRepeatTrue))
    curLoc = 0
    for r in range(nRepeatTrue):
      targetIDs = rowIDs[curLoc:curLoc+L]
      bigResp[targetIDs, r*Ktrue:(r+1)*Ktrue] = resp[targetIDs,:]
      curLoc += L
    resp = bigResp
  elif initname == 'trueparams':
    hmodel.set_global_params(**Data.TrueParams)
    return
  else:
    raise NotImplementedError('Unknown initname: %s' % (initname))

  if hmodel.obsModel.__class__.__name__.count('Gauss') > 0:
    LP = dict(resp=resp)
  else:
    LP = FromScratchMult.getLPfromResp(resp, Data)
  SS = hmodel.get_global_suff_stats(Data, LP)
  hmodel.update_global_params(SS)



def calc_resp_from_true_labels(Data):
  TrueLabels = Data.TrueLabels
  uniqueLabels = np.unique(TrueLabels)
  Ktrue = len(uniqueLabels)
  resp = np.zeros((Data.nObs, Ktrue))
  for k in range(Ktrue):
    mask = TrueLabels == uniqueLabels[k]
    resp[mask,k] = 1.0
  return resp