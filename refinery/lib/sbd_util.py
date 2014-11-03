import re, cPickle, os, gzip, sys, math

def save_pickle(data, path):
    o = gzip.open(path, 'wb')
    cPickle.dump(data, o)
    o.close()

def load_pickle(path):
    i = gzip.open(path, 'rb')
    data = cPickle.load(i)
    i.close()
    return data

def die(msg):
    print '\nERROR: %s' %msg
    sys.exit()

def logit(x, y=1):
    return 1.0 / (1 + math.e ** (-1*y*x))

def get_files(path, pattern):
    """
    Recursively find all files rooted in <path> that match the regexp <pattern>
    """
    L = []
    
    # base case: path is just a file
    if (re.match(pattern, os.path.basename(path)) != None) and os.path.isfile(path):
        L.append(path)
        return L

    # general case
    if not os.path.isdir(path):
        return L

    contents = os.listdir(path)
    for item in contents:
        item = path + item
        if (re.search(pattern, os.path.basename(item)) != None) and os.path.isfile(item):
            L.append(item)
        elif os.path.isdir(path):
            L.extend(get_files(item + '/', pattern))

    return L

class Counter(dict):

   def __getitem__(self, entry):
       try:
           return dict.__getitem__(self, entry)
       except KeyError:
           return 0.0

   def copy(self):
       return Counter(dict.copy(self))

   def __add__(self, counter):
       """
       Add two counters together in obvious manner.
       """
       newCounter = Counter()
       for entry in set(self).union(counter):
           newCounter[entry] = self[entry] + counter[entry]
       return newCounter

   def sortedKeys(self):
       """
       returns a list of keys sorted by their values
       keys with the highest values will appear first
       """
       sortedItems = self.items()
       compare = lambda x,y: sign(y[1] - x[1])
       sortedItems.sort(cmp=compare)
       return [x[0] for x in sortedItems]

   def totalCount(self):
       """
       returns the sum of counts for all keys
       """
       return sum(self.values())

   def incrementAll(self, value=1):
       """
       increment all counts by value
       helpful for removing 0 probs
       """
       for key in self.keys():
           self[key] += value

   def display(self):
       """
       a nicer display than the built-in dict.__repr__
       """
       for key, value in self.items():
           s = str(key) + ': ' + str(value)
           print s

   def displaySorted(self, N=10):
       """
       display sorted by decreasing value
       """
       sortedKeys = self.sortedKeys()
       for key in sortedKeys[:N]:
           s = str(key) + ': ' + str(self[key])
           print s

def normalize(counter):
   """
   normalize a counter by dividing each value by the sum of all values
   """
   counter = Counter(counter)
   normalizedCounter = Counter()
   total = float(counter.totalCount())
   for key in counter.keys():
       value = counter[key]
       normalizedCounter[key] = value / total
   return normalizedCounter
