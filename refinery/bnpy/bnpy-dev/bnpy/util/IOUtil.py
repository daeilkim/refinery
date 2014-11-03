import numpy as np

def flatstr2np( xvecstr ):
  return np.asarray( [float(x) for x in xvecstr.split()] )

def np2flatstr( X, fmt="% .6f" ):
  return ' '.join( [fmt%(x) for x in np.asarray(X).flatten() ] )  
