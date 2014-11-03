'''
VerificationUtil.py

Verification utilities, for checking whether numerical variables are "equal".
'''
import numpy as np

def isEvenlyDivisibleFloat(a, b, margin=1e-6):
  ''' Returns true/false for whether a is evenly divisible by b 
        within a (small) numerical tolerance
      Examples
      --------
      >>> isEvenlyDivisibleFloat( 1.5, 0.5)
      True
      >>> isEvenlyDivisibleFloat( 1.0, 1./3)
      True
  '''
  cexact = np.asarray(a)/float(b)
  cround = np.round(cexact)
  return abs(cexact - cround) < margin
  
def closeAtMSigFigs(A, B, M=10, tol=5):
  ''' Returns true/false for whether A and B are numerically "close"
          aka roughly equal at M significant figures

      Only makes sense for numbers on scale of abs. value 1.0 or larger.      
      Log evidences will usually always be at this scale.

      Examples
      --------
      >>> closeAtMSigFigs(1234, 1000, M=1)  # margin is 500 
      True
      >>> closeAtMSigFigs(1234, 1000, M=2)  # margin is 50 
      False
      >>> closeAtMSigFigs(1034, 1000, M=2)  # margin is 50 
      True
      >>> closeAtMSigFigs(1005, 1000, M=3)  # margin is 5 
      True

      >>> closeAtMSigFigs(44.5, 49.5, M=1) # margin is 5 
      True
      >>> closeAtMSigFigs(44.5, 49.501, M=1) # just over the margin
      False
      >>> closeAtMSigFigs(44.499, 49.5, M=1) 
      False
  '''
  A = float(A)
  B = float(B)
  # Enforce abs(A) >= abs(B)
  if abs(A) < abs(B):
    tmp = A
    A = B
    B = tmp
  assert abs(A) >= abs(B)

  # Find the scale that A (the larger of the two) possesses
  #  A ~= 10 ** (P10)
  P10 = int(np.floor(np.log10(abs(A))))

  # Compare the difference between A and B
  #   to the allowed margin THR
  diff = abs(A - B)
  if P10 >= 0:
    THR = tol * 10.0**(P10 - M)
    THR = (1 + 1e-11) * THR 
    # make THR just a little bigger to avoid issues where 2.0 and 1.95
    # aren't equal at 0.05 margin due to rounding errors
    return np.sign(A) == np.sign(B) and diff <= THR
  else:
    THR = tol * 10.0**(-M)
    THR = (1 + 1e-11) * THR
    return diff <= THR

