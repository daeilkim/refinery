'''
Bars2D.py

Generic functions for creating toy bars data
'''
import numpy as np

def Create2DBarsTopicWordParams(V, K, fracMassOnTopic=0.95, PRNG=np.random):
  ''' Create parameters of each topics distribution over words
      
      Args
      ---------
      V : int vocab size
      K : int number of topics
      fracMassOnTopic : fraction of total probability mass for "on-topic" words
      PRNG : random number generator (for reproducibility)

      Returns
      ---------
      topics : K x V matrix, real positive numbers whose rows sum to one
  '''
  sqrtV = int(np.sqrt(V))
  BarWidth = sqrtV/ (K/2) # number of consecutive words in each bar
  B = V/ (K/2) # total number of "on topic" words in each bar

  topics = np.zeros((K,V))
  # Make horizontal bars
  for k in range(K/2):
    wordIDs = range(B*k, B*(k+1))
    topics[k, wordIDs] = 1.0

  # Make vertical bars
  for k in range(K/2):
    wordIDs = list()
    for b in range(sqrtV):
      start = b * sqrtV + k*BarWidth
      wordIDs.extend( range(start, start+BarWidth))
    topics[K/2 + k, wordIDs] = 1.0

  # Add smoothing mass to all entries in "topics"
  #  instead of picking this value out of thin air, instead,
  #  set it so that 95% of the mass of each topic is on the "on-topic" bar words
  #  if s is the smoothing mass added, and B is num "on topic" words, then
  #   fracMassOnTopic = (1 + s) * B / ( (1+s)*B + s*(V-B) ), and we solve for s
  smoothMass = (1 - fracMassOnTopic)/(fracMassOnTopic*V - B)*B
  topics += (2 * smoothMass) * PRNG.rand(K,V)

  # Ensure each row of topics is a probability vector
  for k in xrange(K):
    topics[k,:] /= np.sum(topics[k,:])

  assert np.sum(topics[0, :B]) > fracMassOnTopic - 0.05
  assert np.sum(topics[1, B:2*B]) > fracMassOnTopic - 0.05
  assert np.sum(topics[-1, wordIDs]) > fracMassOnTopic - 0.05
  return topics