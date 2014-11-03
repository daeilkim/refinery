import DeadLeaves as DL

DL.makeTrueParams(25)

def get_data(**kwargs):
  return DL.get_data(**kwargs)
  
def get_minibatch_iterator(**kwargs):
  return DL.get_minibatch_iterator(**kwargs)
  
def get_short_name():
  return DL.get_short_name()

def get_data_info():
  return DL.get_data_info()
  

if __name__ == '__main__':
  DL.plotTrueCovMats(doShowNow=False)
  DL.plotImgPatchPrototypes()

  
