'''
'''
import numpy as np
import scipy.io
import os
from distutils.dir_util import mkpath

def makePrefixForLap(lap):
  return 'Lap%08.3f' % (lap)

def save_model(hmodel, fname, prefix, doSavePriorInfo=True, doLinkBest=False):
  ''' saves HModel object to mat file persistently
      
      Args
      --------
      hmodel: HModel to save
      fname: absolute full path of directory to save in
      prefix: prefix for file name, like 'Iter00004' or 'Best'
      doSavePriorInfo: whether to save prior info
  '''
  if not os.path.exists( fname):
    mkpath( fname )
  save_alloc_model( hmodel.allocModel, fname, prefix, doLinkBest=doLinkBest )
  save_obs_model( hmodel.obsModel, fname, prefix, doLinkBest=doLinkBest )
  if doSavePriorInfo:
    save_alloc_prior( hmodel.allocModel, fname)
    save_obs_prior( hmodel.obsModel, fname)
    
def save_alloc_model(amodel, fpath, prefix, doLinkBest=False):
  amatname = prefix + 'AllocModel.mat'
  outmatfile = os.path.join( fpath, amatname )
  adict = amodel.to_dict()
  adict.update( amodel.to_dict_essential() )
  scipy.io.savemat( outmatfile, adict, oned_as='row')
  if doLinkBest and prefix != 'Best':
    create_best_link( outmatfile, os.path.join(fpath,'BestAllocModel.mat'))
          
def save_obs_model(obsmodel, fpath, prefix, doLinkBest=False):  
  amatname = prefix + 'ObsModel.mat'
  outmatfile = os.path.join( fpath, amatname )
  compList = list()
  for k in xrange( obsmodel.K ):
    compList.append( obsmodel.comp[k].to_dict() )
  myDict = obsmodel.to_dict_essential()
  for key in compList[0].keys():
    if key in myDict:
      continue
    myDict[key] = np.squeeze(np.dstack([ compDict[key] for compDict in compList]))
  scipy.io.savemat( outmatfile, myDict, oned_as='row')
  if doLinkBest and prefix != 'Best':
    create_best_link( outmatfile, os.path.join(fpath,'BestObsModel.mat'))    
  
def save_alloc_prior( amodel, fpath):
  outpath = os.path.join( fpath, 'AllocPrior.mat')
  adict = amodel.get_prior_dict()
  if len( adict.keys() ) == 0:
    return None
  scipy.io.savemat( outpath, adict, oned_as='row')

def save_obs_prior( obsModel, fpath):
  outpath = os.path.join( fpath, 'ObsPrior.mat')
  adict = obsModel.get_prior_dict()
  if len( adict.keys() ) == 0:
    return None
  scipy.io.savemat( outpath, adict, oned_as='row')

def create_best_link( hardmatfile, linkmatfile):
  ''' Creates a symlink file named linkmatfile that points to hardmatfile,
      where both are full valid absolute file system paths 
  '''
  if os.path.islink( linkmatfile):
    os.unlink( linkmatfile )
  if os.path.exists(linkmatfile):
    os.remove(linkmatfile)
  if os.path.exists( hardmatfile ):
    os.symlink( hardmatfile, linkmatfile )
