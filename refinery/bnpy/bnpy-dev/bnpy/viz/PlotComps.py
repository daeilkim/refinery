'''
PlotComps.py

Executable for plotting learned parameters for each component

Usage (command-line)
-------
python -m bnpy.viz.PlotComps dataName aModelName obsModelName algName [kwargs]

'''
from matplotlib import pylab
import numpy as np
import argparse
import os
import sys
import bnpy
import bnpy.ioutil.BNPYArgParser as BNPYArgParser
from bnpy.ioutil import ModelReader

def main():
  args = parse_args()
  jobpath, taskids = parse_jobpath_and_taskids(args)

  for taskid in taskids:
    taskpath = os.path.join(jobpath, taskid)
    if args.lap is not None:
      prefix, bLap = ModelReader.getPrefixForLapQuery(taskpath, args.lap)      
      if bLap != args.lap:
        print 'Using saved lap: ', bLap
    else:
      prefix = 'Best' # default

    hmodel = ModelReader.load_model(taskpath, prefix)
    plotModelInNewFigure(jobpath, hmodel, args)
    if args.savefilename is not None:
      pylab.show(block=False)
      pylab.savefig(args.savefilename % (taskid))
  
  if args.savefilename is None:
    pylab.show(block=True)
        
def plotModelInNewFigure(jobpath, hmodel, args):
  figHandle = pylab.figure()
  if args.doPlotData:
    Data = loadData(jobpath)
    plotData(Data)

  if hmodel.getObsModelName().count('ZMGauss') and hmodel.obsModel.D > 2:
    bnpy.viz.GaussViz.plotCovMatFromHModel(hmodel)
  elif hmodel.getObsModelName().count('Gauss'):
    bnpy.viz.GaussViz.plotGauss2DFromHModel(hmodel)
  elif args.dataName.lower().count('bars') > 0:
    pylab.close(figHandle)
    if args.doPlotTruth:
      Data = loadData(jobpath)
    else:
      Data = None
    bnpy.viz.BarsViz.plotBarsFromHModel(hmodel, Data=Data, 
                                        sortBySize=args.doSort, doShowNow=False)
  else:
    raise NotImplementedError('Unrecognized data/obsmodel combo')

def plotData(Data, nObsPlot=5000):
  ''' Plot data items, at most nObsPlot distinct points (for quick rendering)
  '''
  if type(Data) == bnpy.data.XData:
    PRNG = np.random.RandomState(nObsPlot)
    pIDs = PRNG.permutation(Data.nObs)[:nObsPlot]
    if Data.dim > 1:
      pylab.plot(Data.X[pIDs,0], Data.X[pIDs,1], 'k.')  
    else:
      hist, bin_edges = pylab.histogram(Data.X, bins=25)
      xs = bin_edges[:-1]
      ys = np.asarray(hist, dtype=np.float32) / np.sum(hist)
      pylab.bar(xs, ys, width=0.8*(bin_edges[1]-bin_edges[0]), color='k')

def loadData(jobpath):
  ''' Load in bnpy Data obj associated with given learning task.
  '''
  bnpyoutdir = os.environ['BNPYOUTDIR']
  subdirpath = jobpath[len(bnpyoutdir):]
  fields = subdirpath.split(os.path.sep)
  dataname = fields[0]
  sys.path.append(os.environ['BNPYDATADIR'])
  datamodulepath = os.path.join(os.environ['BNPYDATADIR'], dataname+".py")
  if not os.path.exists(datamodulepath):
    return None
    #raise ValueError("Could not find data %s" % (dataname))
  datamod = __import__(dataname, fromlist=[])
  return datamod.get_data()
  
  
def parse_args():
  ''' Parse cmd line arguments
  '''
  parser = argparse.ArgumentParser() 
   
  BNPYArgParser.addRequiredVizArgsToParser(parser)
  BNPYArgParser.addStandardVizArgsToParser(parser)
  parser.add_argument('--lap', default=None, type=float,
        help="Specific lap at which to plot parameters." \
             + " If exact lap not available, instead plots nearest lap.")
  parser.add_argument('--doPlotData', action='store_true', default=False,
        help="If present, also plot training data.")
  parser.add_argument('--doPlotTruth', action='store_true', default=False,
        help="If present, also plot true model params that generated data.")
  parser.add_argument('--doSort', action='store_true', default=False,
        help="If present, sort parameters by global appearance probabilities.")
  args = parser.parse_args()
  return args

def parse_jobpath_and_taskids(args):
  rootpath = os.path.join(os.environ['BNPYOUTDIR'], args.dataName, 
                              args.allocModelName, args.obsModelName)
  jobpath = os.path.join(rootpath, args.algNames, args.jobnames)
  if not os.path.exists(jobpath):
    raise ValueError("No such path: %s" % (jobpath))
  taskids = BNPYArgParser.parse_task_ids(jobpath, args.taskids)

  # Verify that the intended savefile will work as expected!
  if args.savefilename is not None:
    if args.savefilename.count('%') and len(taskids) > 1:
      try:
        args.savefilename % ('1')
      except TypeError:
        raise ValueError("Missing or bad format string in savefilename %s" %  
                        (args.savefilename)
                      )  
  return jobpath, taskids


if __name__ == "__main__":
  main()
