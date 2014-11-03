'''
PlotELBO.py

Executable for plotting the learning objective function (log evidence)
  vs. time/number of passes thru data (laps)

Usage (command-line)
-------
python -m bnpy.viz.PlotELBO dataName aModelName obsModelName algName [kwargs]
'''
from matplotlib import pylab
import numpy as np
import argparse
import os
import bnpy.ioutil.BNPYArgParser as BNPYArgParser
import bnpy
import matplotlib
matplotlib.rcParams['text.usetex'] = False

Colors = [(0,0,0), # black
          (0,0,1), # blue
          (1,0,0), # red
          (0,1,0.25), # green (darker)
          (1,0,1), # magenta
          (0,1,1), # cyan
          (1,0.6,0), #orange
         ]
XLabelMap = dict(laps='num pass thru data',
                  iters='num steps in alg',
                  times='elapsed time (sec)'
                  )  
YMin = None
YMax = None      
     
def main():
  args = parse_args()
  for (jobpath, jobname, color) in job_to_plot_generator(args):
    plot_all_tasks_for_job(jobpath, args, jobname, color)
  pylab.legend(loc='best')  
  if args.savefilename is not None:
    pylab.show(block=False)
    pylab.savefig(args.savefilename)
  else:
    pylab.show(block=True)
        

def plot_all_tasks_for_job(jobpath, args, jobname=None, color=None):
  ''' Create line plot in current matplotlib figure
      for each task/run of the designated jobpath
  '''
  if not os.path.exists(jobpath):
    raise ValueError("No such path: %s" % (jobpath))
  
  taskids = BNPYArgParser.parse_task_ids(jobpath, args.taskids)
    
  xAll = list()
  yAll = list()
  xLocs = list()
  yLocs = list()
  for tt, taskid in enumerate(taskids):
    xs = np.loadtxt(os.path.join(jobpath, taskid, args.xvar+'.txt'))
    ys = np.loadtxt(os.path.join(jobpath, taskid, 'evidence.txt'))
    # remove first-lap of moVB, since ELBO is not accurate
    if jobpath.count('moVB') > 0 and args.xvar == 'laps':
      mask = xs >= 1.0
      xs = xs[mask]
      ys = ys[mask]
    if args.traceEvery is not None:
      mask = bnpy.util.isEvenlyDivisibleFloat(xs, args.traceEvery)
      xs = xs[mask]
      ys = ys[mask]


    plotargs = dict(markersize=10, linewidth=2, label=None,
                    color=color, markeredgecolor=color)
    if tt == 0:
      plotargs['label'] = jobname
    pylab.plot(xs, ys, '.-', **plotargs)
    if len(ys) > 0:
      xLocs.append(xs[-1])
      yLocs.append(ys[-1])
      yAll.extend(ys[1:])
      xAll.extend(xs[1:])
      
  # Zoom in to the useful part of the ELBO trace
  if len(yAll) > 0:
    global YMin, YMax
    ymin = np.percentile(yAll, 1)
    ymax = np.max(yAll)
    if YMin is None:
      YMin = ymin
      YMax = ymax
    else:
      YMin = np.minimum(ymin, YMin)
      YMax = np.maximum(YMax, ymax)
    blankmargin = 0.08*(YMax - YMin)
    pylab.ylim( [YMin, YMax + blankmargin])
  pylab.xlabel(XLabelMap[args.xvar])
  pylab.ylabel('log evidence')
   
def job_to_plot_generator(args):
  ''' Generates tuples (jobpath, jobname, color), each specifies a line plot
  '''
  rootpath = os.path.join(os.environ['BNPYOUTDIR'], args.dataName, \
                              args.allocModelName, args.obsModelName)
  cID = 0
  for algname in args.algNames:
    for jobname in args.jobnames:
      curjobpath = os.path.join(rootpath, algname, jobname)
      if not os.path.exists(curjobpath):
        print 'DOES NOT EXIST:', curjobpath
        continue
      cID += 1
      if args.legendnames is not None:
        jobname = args.legendnames[cID - 1]
      yield curjobpath, jobname, Colors[cID]
  
def parse_args():
  ''' Returns Namespace of parsed arguments retrieved from command line
  '''
  parser = argparse.ArgumentParser()
  BNPYArgParser.addRequiredVizArgsToParser(parser)
  BNPYArgParser.addStandardVizArgsToParser(parser)
  parser.add_argument('--xvar', type=str, default='laps',
        help="name of x axis variable to plot. one of {iters,laps,times}")

  parser.add_argument('--traceEvery', type=str, default=None,
        help="Specifies how often to plot data points. For example, traceEvery=10 only plots data points associated with laps divisible by 10.")
  parser.add_argument('--legendnames', type=str, default=None,
        help="optional names to show on legend in place of jobnames")
  args = parser.parse_args()
  args.algNames = args.algNames.split(',')
  args.jobnames = args.jobnames.split(',')
  if args.legendnames is not None:
    args.legendnames = args.legendnames.split(',')
    #assert len(args.legendnames) == len(args.jobnames) * len(args.algNames)
  return args

if __name__ == "__main__":
  main()

