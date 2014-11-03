'''
SuffStatBag.py

Container object for sufficient statistics in bnpy models.
Uses ParamBag as internal representation.

Tracks three possible sets of parameters, each with own ParamBag
* sufficient statistics fields
* (optional) precomputed ELBO terms
* (optional) precomputed terms for potential merges

'''
import copy
import numpy as np
from ParamBag import ParamBag

class SuffStatBag(object):
  def __init__(self, K=0, D=0):
    self._Fields = ParamBag(K=K, D=D)

  def copy(self):
    return copy.deepcopy(self)

  def setField(self, key, value, dims=None):
    self._Fields.setField(key, value, dims=dims)

  def setELBOFieldsToZero(self):
    if self.hasELBOTerms():
      self._ELBOTerms.setAllFieldsToZero()

  def setMergeFieldsToZero(self):
    if self.hasMergeTerms():
      self._MergeTerms.setAllFieldsToZero()

  # ======================================================= Amp factor
  def hasAmpFactor(self):
    return hasattr(self, 'ampF')
        
  def applyAmpFactor(self, ampF):
    self.ampF = ampF
    for key in self._Fields._FieldDims:
      arr = getattr(self._Fields, key)
      if arr.ndim == 0:
        # Edge case: in-place updates don't work with de-referenced 0-d arrays
        setattr(self._Fields, key, arr * ampF)
      else:
        arr *= ampF
      
  # ======================================================= ELBO terms
  def hasELBOTerms(self):
    return hasattr(self, '_ELBOTerms')

  def hasELBOTerm(self, key):
    if not hasattr(self, '_ELBOTerms'):
      return False
    return hasattr(self._ELBOTerms, key)

  def getELBOTerm(self, key):
    return getattr(self._ELBOTerms, key)

  def setELBOTerm(self, key, value, dims=None):
    if not hasattr(self, '_ELBOTerms'):
      self._ELBOTerms = ParamBag(K=self.K)
    self._ELBOTerms.setField(key, value, dims=dims)

  # ======================================================= ELBO merge terms
  def hasMergeTerms(self):
    return hasattr(self, '_MergeTerms')

  def hasMergeTerm(self, key):
    if not hasattr(self, '_MergeTerms'):
      return False
    return hasattr(self._MergeTerms, key)

  def getMergeTerm(self, key):
    return getattr(self._MergeTerms, key)

  def setMergeTerm(self, key, value, dims=None):
    if not hasattr(self, '_MergeTerms'):
      self._MergeTerms = ParamBag(K=self.K)
    self._MergeTerms.setField(key, value, dims=dims)


  # ======================================================= Merge comps
  def mergeComps(self, kA, kB):
    ''' Merge components kA, kB into a single component
    '''
    if self.K <= 1:
      raise ValueError('Must have at least 2 components to merge.')
    if kB == kA:
      raise ValueError('Distinct component ids required.')
    for key, dims in self._Fields._FieldDims.items():
      if dims is not None and dims != ():
        arr = getattr(self._Fields, key)
        if self.hasMergeTerm(key) and dims == ('K'):
          # some special terms need to be precomputed, like sumLogPiActive
          arr[kA] = getattr(self._MergeTerms, key)[kA,kB]
        else:
          # applies to vast majority of all fields
          arr[kA] += arr[kB]

    if self.hasELBOTerms():
      for key, dims in self._ELBOTerms._FieldDims.items():
        if self.hasMergeTerm(key) and dims == ('K'):
          arr = getattr(self._ELBOTerms, key)
          mArr = getattr(self._MergeTerms, key)
          arr[kA] = mArr[kA,kB]

    if self.hasMergeTerms():
      for key, dims in self._MergeTerms._FieldDims.items():
        if dims == ('K','K'):
          mArr = getattr(self._MergeTerms, key)
          mArr[kA,kA+1:] = np.nan
          mArr[:kA,kA] = np.nan

    self._Fields.removeComp(kB)
    if self.hasELBOTerms():
      self._ELBOTerms.removeComp(kB)
    if self.hasMergeTerms():
      self._MergeTerms.removeComp(kB)

  # ======================================================= Insert comps
  def insertComps(self, SS):
    self._Fields.insertComps(SS)
    if hasattr(self, '_ELBOTerms'):
      self._ELBOTerms.insertEmptyComps(SS.K)
    if hasattr(self, '_MergeTerms'):
      self._MergeTerms.insertEmptyComps(SS.K)

  def insertEmptyComps(self, Kextra):
    self._Fields.insertEmptyComps(Kextra)
    if hasattr(self, '_ELBOTerms'):
      self._ELBOTerms.insertEmptyComps(Kextra)
    if hasattr(self, '_MergeTerms'):
      self._MergeTerms.insertEmptyComps(Kextra)

  # ======================================================= Remove comp
  def removeComp(self, k):
    self._Fields.removeComp(k)
    if hasattr(self, '_ELBOTerms'):
      self._ELBOTerms.removeComp(k)
    if hasattr(self, '_MergeTerms'):
      self._MergeTerms.removeComp(k)

  # ======================================================= Get comp
  def getComp(self, k, doCollapseK1=True):
    SS = SuffStatBag(K=1, D=self.D)
    SS._Fields = self._Fields.getComp(k, doCollapseK1=doCollapseK1)
    return SS

  # ======================================================= Override add
  def __add__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    SSsum = SuffStatBag(K=self.K, D=self.D)
    SSsum._Fields = self._Fields + PB._Fields
    if hasattr(self, '_ELBOTerms'):
      SSsum._ELBOTerms = self._ELBOTerms + PB._ELBOTerms
    if hasattr(self, '_MergeTerms'):
      SSsum._MergeTerms = self._MergeTerms + PB._MergeTerms
    return SSsum

  def __iadd__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    self._Fields += PB._Fields
    if hasattr(self, '_ELBOTerms'):
      self._ELBOTerms += PB._ELBOTerms
    if hasattr(self, '_MergeTerms'):
      self._MergeTerms += PB._MergeTerms
    return self

  # ======================================================= Override subtract
  def __sub__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    SSsum = SuffStatBag(K=self.K, D=self.D)
    SSsum._Fields = self._Fields - PB._Fields
    if hasattr(self, '_ELBOTerms'):
      SSsum._ELBOTerms = self._ELBOTerms - PB._ELBOTerms
    if hasattr(self, '_MergeTerms'):
      SSsum._MergeTerms = self._MergeTerms - PB._MergeTerms
    return SSsum

  def __isub__(self, PB):
    if self.K != PB.K or self.D != PB.D:
      raise ValueError('Dimension mismatch')
    self._Fields -= PB._Fields
    if hasattr(self, '_ELBOTerms'):
      self._ELBOTerms -= PB._ELBOTerms
    if hasattr(self, '_MergeTerms'):
      self._MergeTerms -= PB._MergeTerms
    return self

  # ======================================================= Override getattr
  def __getattr__(self, key):
    if key == "_Fields":
      return object.__getattribute__(self, key)
    elif hasattr(self._Fields, key):
      return getattr(self._Fields,key)
    elif key == '__deepcopy__': # workaround to allow copying
      return None
    elif key in self.__dict__:
      return self.__dict__[key]
    # Field named 'key' doesnt exist. 
    errmsg = "'SuffStatBag' object has no attribute '%s'" % (key)
    raise AttributeError(errmsg)
