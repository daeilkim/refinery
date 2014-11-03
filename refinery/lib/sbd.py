import re, sys, os, math, tempfile, collections, subprocess
import sbd_util, word_tokenize

"""
Utilities for disambiguating sentence boundaries
Copyright Dan Gillick, 2009.

TODO:
- capitalized headlines screw things up?
- deal with ?! maybe just assume these sentence boundaries always
"""

## globals
SVM_LEARN = os.path.abspath("") + "/lib/svmlite/svm_learn"
SVM_CLASSIFY = os.path.abspath("") + "/lib/svmlite/svm_classify"

def get_open_fds():
    '''
    return the number of open file descriptors for current process

    .. warning: will only work on UNIX-like os-es.
    '''
    pid = os.getpid()
    procs = subprocess.check_output( 
        [ "lsof", '-w', '-Ff', "-p", str( pid ) ] )

    nprocs = len( 
        filter( 
            lambda s: s and s[ 0 ] == 'f' and s[1: ].isdigit(),
            procs.split( '\n' ) )
        )
    return nprocs

def unannotate(t):
    """
    get rid of a tokenized word's annotations
    """
    t = re.sub('(<A>)?(<E>)?(<S>)?$', '', t)
    return t

def clean(t):
    """
    normalize numbers, discard some punctuation that can be ambiguous
    """
    t = re.sub('[.,\d]*\d', '<NUM>', t)
    t = re.sub('[^a-zA-Z0-9,.;:<>\-\'\/?!$% ]', '', t)
    t = t.replace('--', ' ') # sometimes starts a sentence... trouble
    return t
            
def get_features(frag, model):
    """
    ... w1. (sb?) w2 ...
    Features, listed roughly in order of importance:

    (1) w1: word that includes a period
    (2) w2: the next word, if it exists
    (3) w1length: number of alphabetic characters in w1
    (4) w2cap: true if w2 is capitalized
    (5) both: w1 and w2
    (6) w1abbr: log count of w1 in training without a final period
    (7) w2lower: log count of w2 in training as lowercased
    (8) w1w2upper: w1 and w2 is capitalized
    """
    words1 = clean(frag.tokenized).split()
    if not words1: w1 = ''
    else: w1 = words1[-1]
    if frag.next:
        words2 = clean(frag.next.tokenized).split()
        if not words2: w2 = ''
        else: w2 = words2[0]
    else:
        words2 = []
        w2 = ''

    c1 = re.sub('(^.+?\-)', '', w1)
    c2 = re.sub('(\-.+?)$', '', w2)

    feats = {}
    
    feats['w1'] = c1
    feats['w2'] = c2
    feats['both'] = c1 + '_' + c2

    len1 = min(10, len(re.sub('\W', '', c1)))
    
    if c1.replace('.','').isalpha():
        feats['w1length'] = str(len1)
        try: feats['w1abbr'] = str(int(math.log(1+model.non_abbrs[c1[:-1]])))
        except: feats['w1abbr'] = str(int(math.log(1)))

    if c2.replace('.','').isalpha():
        feats['w2cap'] = str(c2[0].isupper())
        try: feats['w2lower'] = str(int(math.log(1+model.lower_words[c2.lower()])))
        except: feats['w2lower'] = str(int(math.log(1)))        
        feats['w1w2upper'] = c1 + '_' + str(c2[0].isupper())

    return feats

def is_sbd_hyp(word):
    """
    todo: expand to ?!
    """
    
    if not '.' in word: return False
    c = unannotate(word)
    if c.endswith('.'): return True
    if re.match('.*\.["\')\]]*$', c): return True
    return False
        
def get_data(files, expect_labels=True, tokenize=False, verbose=False):
    """
    load text from files, returning an instance of the Doc class
    doc.frag is the first frag, and each points to the next
    """
    
    if type(files) == type(''): files = [files]
    frag_list = None
    word_index = 0
    frag_index = 0
    curr_words = []
    lower_words, non_abbrs = sbd_util.Counter(), sbd_util.Counter()

    for file in files:
        sys.stderr.write('reading [%s]\n' %file)
        fh = open(file)
        for line in fh:

            ## deal with blank lines
            if (not line.strip()) and frag_list:
                if not curr_words: frag.ends_seg = True
                else:
                    frag = Frag(' '.join(curr_words))
                    frag.ends_seg = True
                    if expect_labels: frag.label = True
                    prev.next = frag
                    if tokenize:
                        tokens = word_tokenize.tokenize(frag.orig)
                    frag.tokenized = tokens
                    frag_index += 1
                    prev = frag
                    curr_words = []

            for word in line.split():
                curr_words.append(word)

                if is_sbd_hyp(word):
                #if True: # hypothesize all words
                    frag = Frag(' '.join(curr_words))
                    if not frag_list: frag_list = frag
                    else: prev.next = frag
                    
                    ## get label; tokenize
                    if expect_labels: frag.label = int('<S>' in word)
                    if tokenize:
                        tokens = word_tokenize.tokenize(frag.orig)
                    else: tokens = frag.orig
                    tokens = re.sub('(<A>)|(<E>)|(<S>)', '', tokens)
                    frag.tokenized = tokens
                    
                    frag_index += 1
                    prev = frag
                    curr_words = []

                word_index += 1
        fh.close()

        ## last frag
        frag = Frag(' '.join(curr_words))
        if not frag_list: frag_list = frag
        else: prev.next = frag
        if expect_labels: frag.label = int('<S>' in word)
        if tokenize:
            tokens = word_tokenize.tokenize(frag.orig)
        else: tokens = frag.orig
        tokens = re.sub('(<A>)|(<E>)|(<S>)', '', tokens)
        frag.tokenized = tokens
        frag.ends_seg = True
        frag_index += 1

    if verbose: sys.stderr.write(' words [%d] sbd hyps [%d]\n' %(word_index, frag_index))

    ## create a Doc object to hold all this information
    doc = Doc(frag_list)
    return doc

def get_text_data(text, expect_labels=True, tokenize=False, verbose=False):
    """
    get text, returning an instance of the Doc class
    doc.frag is the first frag, and each points to the next
    """
    
    frag_list = None
    word_index = 0
    frag_index = 0
    curr_words = []
    lower_words, non_abbrs = sbd_util.Counter(), sbd_util.Counter()

    for line in text.splitlines():

        ## deal with blank lines
        if (not line.strip()) and frag_list:
            if not curr_words: frag.ends_seg = True
            else:
                frag = Frag(' '.join(curr_words))
                frag.ends_seg = True
                if expect_labels: frag.label = True
                prev.next = frag
                if tokenize:
                    tokens = word_tokenize.tokenize(frag.orig)
                frag.tokenized = tokens
                frag_index += 1
                prev = frag
                curr_words = []

        for word in line.split():
            curr_words.append(word)

            if is_sbd_hyp(word):
                frag = Frag(' '.join(curr_words))
                if not frag_list: frag_list = frag
                else: prev.next = frag
                
                ## get label; tokenize
                if expect_labels: frag.label = int('<S>' in word)
                if tokenize:
                    tokens = word_tokenize.tokenize(frag.orig)
                else: tokens = frag.orig
                tokens = re.sub('(<A>)|(<E>)|(<S>)', '', tokens)
                frag.tokenized = tokens
                
                frag_index += 1
                prev = frag
                curr_words = []
                
            word_index += 1

    ## last frag
    frag = Frag(' '.join(curr_words))
    if not frag_list: frag_list = frag
    else: prev.next = frag
    if expect_labels: frag.label = int('<S>' in word)
    if tokenize:
        tokens = word_tokenize.tokenize(frag.orig)
    else: tokens = frag.orig
    tokens = re.sub('(<A>)|(<E>)|(<S>)', '', tokens)
    frag.tokenized = tokens
    frag.ends_seg = True
    frag_index += 1
        
    if verbose: sys.stderr.write(' words [%d] sbd hyps [%d]\n' %(word_index, frag_index))

    ## create a Doc object to hold all this information
    doc = Doc(frag_list)
    return doc


class Model:
    """
    Abstract Model class holds all relevant information, and includes
    train and classify functions
    """
    def __init__(self, path):
        self.feats, self.lower_words, self.non_abbrs = {}, {}, {}
        self.path = path

    def prep(self, doc):
        self.lower_words, self.non_abbrs = doc.get_stats(verbose=False)
        self.lower_words = dict(self.lower_words)
        self.non_abbrs = dict(self.non_abbrs)

    def train(self, doc):
        abstract

    def classify(self, doc, verbose=False):
        abstract

    def save(self):
        """
        save model objects in self.path
        """
        sbd_util.save_pickle(self.feats, self.path + 'feats')
        sbd_util.save_pickle(self.lower_words, self.path + 'lower_words')
        sbd_util.save_pickle(self.non_abbrs, self.path + 'non_abbrs')

    def load(self):
        """
        load model objects from p
        """
        self.feats = sbd_util.load_pickle(self.path + 'feats')
        self.lower_words = sbd_util.load_pickle(self.path + 'lower_words')
        self.non_abbrs = sbd_util.load_pickle(self.path + 'non_abbrs')

        
class NB_Model(Model):
    """
    Naive Bayes model, with a few tweaks:
    - all feature types are pooled together for normalization (this might help
      because the independence assumption is so broken for our features)
    - smoothing: add 0.1 to all counts
    - priors are modified for better performance (this is mysterious but works much better)
    """

    def train(self, doc):

        sys.stderr.write('training nb... ')
        feats = collections.defaultdict(sbd_util.Counter)
        totals = sbd_util.Counter()

        frag = doc.frag
        while frag:
            for feat, val in frag.features.items():
                feats[frag.label][feat + '_' + val] += 1
            totals[frag.label] += len(frag.features)
            frag = frag.next

        ## add-1 smoothing and normalization
        sys.stderr.write('smoothing... ')
        smooth_inc = 0.1
        all_feat_names = set(feats[True].keys()).union(set(feats[False].keys()))
        for label in [0,1]:
            totals[label] += (len(all_feat_names) * smooth_inc)
            for feat in all_feat_names:
                feats[label][feat] += smooth_inc
                feats[label][feat] /= totals[label]
                self.feats[(label, feat)] = feats[label][feat]
            feats[label]['<prior>'] = totals[label] / totals.totalCount()
            self.feats[(label, '<prior>')] = feats[label]['<prior>']

        sys.stderr.write('done!\n')

    def classify_nb_one(self, frag):
        ## the prior is weird, but it works better this way, consistently
        probs = sbd_util.Counter([(label, self.feats[label, '<prior>']**4) for label in [0,1]])
        for label in probs:
            for feat, val in frag.features.items():
                key = (label, feat + '_' + val)
                if not key in self.feats: continue
                probs[label] *= self.feats[key]

        probs = sbd_util.normalize(probs)
        return probs[1]

    def classify(self, doc, verbose=False):
        if verbose: sys.stderr.write('NB classifying... ')
        frag = doc.frag
        while frag:
            pred = self.classify_nb_one(frag)
            frag.pred = pred
            frag = frag.next
        if verbose: sys.stderr.write('done!\n')

class SVM_Model(Model):
    """
    SVM model (using SVM Light), with a linear kernel, C parameter set to 1
    Non-exhaustive testing of other kernels and parameters showed no improvement
    """

    def train(self, doc):
        """
        takes training data and a path and creates an svm model
        """

        model_file = '%ssvm_model' %self.path

        ## need integer dictionary for features
        sys.stderr.write('training. making feat dict... ')
        feat_list = set()
        frag = doc.frag
        while frag:
            feats = [f+'_'+v for f,v in frag.features.items()]
            for feat in feats: feat_list.add(feat)
            frag = frag.next
        self.feats = dict(zip(feat_list, range(1,len(feat_list)+1)))

        ## training data file
        sys.stderr.write('writing... ')
        lines = []
        frag = doc.frag
        while frag:
            if frag.label == None: sbd_util.die('expecting labeled data [%s]' %frag)
            elif frag.label > 0.5: svm_label = '+1'
            elif frag.label < 0.5: svm_label = '-1'
            else: continue
            line = '%s ' %svm_label
            feats = [f+'_'+v for f,v in frag.features.items()]
            svm_feats = [self.feats[f] for f in feats]
            svm_feats.sort(lambda x,y: x-y)
            line += ' '.join(['%d:1' %x for x in svm_feats])
            lines.append(line)
            frag = frag.next

        unused, train_file = tempfile.mkstemp()
        fh = open(train_file, 'w')
        fh.write('\n'.join(lines) + '\n')
        fh.close()
    
        ## train an svm model
        sys.stderr.write('running svm... ')
        options = '-c 1 -v 0'
        cmd = '%s %s %s %s' %(SVM_LEARN, options, train_file, model_file)
        os.system(cmd)
        sys.stderr.write('done!\n')

        ## clean up
        os.remove(train_file)

    def classify(self, doc, verbose=False):

        model_file = '%ssvm_model' %self.path
        if not self.feats: sbd_util.die('Incomplete model')
        if not os.path.isfile(model_file): sbd_util.die('no model [%s]' %model_file)

        ## testing data file
        if verbose: sys.stderr.write('SVM classifying... ')
        lines = []
        frag = doc.frag
        while frag:
            if frag.label == None: svm_label = '0'
            elif frag.label: svm_label = '+1'
            else: svm_label = '-1'
            line = '%s ' %svm_label
            feats = [f+'_'+v for f,v in frag.features.items()]
            svm_feats = [self.feats[f] for f in feats if f in self.feats]
            svm_feats.sort(lambda x,y: x-y)
            line += ' '.join(['%d:1' %x for x in svm_feats])
            lines.append(line)
            frag = frag.next

        #print "!----!",get_open_fds()
        #unused, test_file = tempfile.mkstemp()
        test_file = "tmp1"
        fh = open(test_file, 'w')
        fh.write('\n'.join(lines) + '\n')
        fh.close()
        #print "!----!",get_open_fds()

    
        #unused, pred_file = tempfile.mkstemp()
        pred_file = "tmp2"
        options = '-v 0'
        cmd = '%s %s %s %s %s' %(SVM_CLASSIFY, options, test_file, model_file, pred_file)
        os.system(cmd)

        ## get predictions
        total = 0
        pf = open(pred_file,'r')
        #print pf
        preds = map(float, pf.read().splitlines())
        frag = doc.frag
        while frag:
            frag.pred = sbd_util.logit(preds[total])
            frag = frag.next
            total += 1

        ## clean up
        pf.close()
        os.remove(test_file)
        os.remove(pred_file)
        
        if verbose: sys.stderr.write('done!\n')
        
class Doc:
    """
    A Document points to the head of a Frag object
    """
    
    def __init__(self, frag):
        self.frag = frag

    def __str__(self):
        s = []
        curr = self.frag
        while curr: s.append(curr)
        return '\n'.join(s)

    def get_stats(self, verbose):
        if verbose: sys.stderr.write('getting statistics... ')
        lower_words = sbd_util.Counter()
        non_abbrs = sbd_util.Counter()
        
        frag = self.frag
        while frag:
            for word in frag.tokenized.split():
                if word.replace('.', '').isalpha():
                    if word.islower(): lower_words[word.replace('.','')] += 1
                    if not word.endswith('.'): non_abbrs[word] += 1
            frag = frag.next

        if verbose: sys.stderr.write('lowercased [%d] non-abbrs [%d]\n'
                                     %(len(lower_words), len(non_abbrs)))

        return lower_words, non_abbrs

    def featurize(self, model, verbose=False):
        if verbose: sys.stderr.write('featurizing... ')
        frag = self.frag
        while frag:
            frag.features = get_features(frag, model)
            frag = frag.next
        if verbose: sys.stderr.write('done!\n')

    def segment(self, use_preds=False, tokenize=False, output=None, list_only=False):
        """
        output all the text, split according to predictions or labels
        """
        sents = []
        thresh = 0.5
        sent = []
        frag = self.frag
        while frag:
            if tokenize: text = frag.tokenized
            else: text = frag.orig
            sent.append(text)
            if frag.ends_seg or (use_preds and frag.pred>thresh) or (not use_preds and frag.label>thresh):
                if not frag.orig: break
                sent_text = ' '.join(sent)
                if frag.ends_seg: spacer = '\n\n'
                else: spacer = '\n'
                if output: output.write(sent_text + spacer)
                elif not list_only: sys.stdout.write(sent_text + spacer)
                sents.append(sent_text)
                sent = []
            frag = frag.next
        return sents

    def show_results(self, verbose=False):

        thresh = 0.5
        total, correct = 0, 0
        frag = self.frag
        while frag:
            total += 1
            if frag.label == (frag.pred > thresh):
                correct += 1
            else:
                w1 = ' '.join(frag.tokenized.split()[-2:])
                if frag.next: w2 = ' '.join(frag.next.tokenized.split()[:2])
                else: w2 = '<EOF>'
                if verbose:
                    print '[%d] [%1.4f] %s?? %s' %(frag.label, frag.pred, w1, w2)

            frag = frag.next

        error = 1 - (1.0 * correct / total)
        print 'correct [%d] total [%d] error [%1.4f]' %(correct, total, error)
        
class Frag:
    """
    A fragment of text that ends with a possible sentence boundary
    """
    def __init__(self, orig):
        self.orig = orig
        self.next = None
        self.ends_seg = False
        self.tokenized = False
        self.pred = None
        self.label = None
        self.features = None

    def __str__(self):
        s = self.orig
        if self.ends_seg: s += ' <EOS> '
        return s
    
def build_model(files, options):

    ## create a Doc object from some labeled data
    train_corpus = get_data(files, tokenize=options.tokenize)

    ## create a new model
    if options.svm: model = SVM_Model(options.model_path)
    else: model = NB_Model(options.model_path)
    model.prep(train_corpus)

    ## featurize the training corpus
    train_corpus.featurize(model, verbose=True)

    ## run the model's training routine
    model.train(train_corpus)

    ## save the model
    model.save()
    return model

def load_sbd_model(model_path = 'model_nb/', use_svm=False):
    sys.stderr.write('loading model from [%s]... ' %model_path)
    if use_svm: model = SVM_Model(model_path)
    else: model = NB_Model(model_path)
    model.load()
    sys.stderr.write('done!\n')
    return model

def sbd_text(model, text, do_tok=True):
    """
    A hook for segmenting text in Python code:

    import sbd
    m = sbd.load_sbd_model('/u/dgillick/sbd/splitta/test_nb/', use_svm=False)
    sents = sbd.sbd_text(m, 'Here is. Some text')
    """

    #print "!!!!!!",get_open_fds()

    data = get_text_data(text, expect_labels=False, tokenize=True)
    #print "!!!!!!",get_open_fds()
    data.featurize(model, verbose=False)
    #print "!!!!!!",get_open_fds()
    model.classify(data, verbose=False)
    #print "!!!!!!",get_open_fds()
    sents = data.segment(use_preds=True, tokenize=do_tok, list_only=True)
    #print "!!???!",get_open_fds()
    return sents


if __name__ == '__main__':

    
    modelpath = '/home/chonger/Downloads/model_svm/'

    model = load_sbd_model(modelpath,True)

    sam = '/home/chonger/Downloads/sample.txt'

    sents = [l for l in open(sam)]

    toseg = ""

    for s in sents:
        if len(s.strip()) > 0:
           print ">",s
           toseg += " " + s.strip()

    print toseg

    ss = sbd_text(model,toseg)

    for s in ss:
        print "!",s

        
    '''
    test = get_data(
    test.featurize(model, verbose=True)
    model.classify(test, verbose=True)
    test.segment(use_preds=True, tokenize=False, output=None)
    '''

def poo():
    

    ## labeled data
    data_root = '/u/dgillick/workspace/sbd/'
    brown_data = data_root + 'whiskey/brown.1'
    wsj_data = data_root + 'whiskey/satz.1'
    poe_data = data_root + 'whiskey/poe.1'
    new_wsj_data = data_root + 'whiskey/wsj.1'

    ## install root
    install_root = '/home/chonger/Downloads/splitta/'

    ## options
    from optparse import OptionParser
    usage = 'usage: %prog [options] <text_file>'
    parser = OptionParser(usage=usage)
    parser.add_option('-v', '--verbose', dest='verbose', default=False,
                      action='store_true', help='verbose output')
    parser.add_option('-t', '--tokenize', dest='tokenize', default=False,
                      action='store_true', help='write tokenized output')
    parser.add_option('-m', '--model', dest='model_path', type='str', default='model_nb',
                      help='model path')
    parser.add_option('-o', '--output', dest='output', type='str', default=None,
                      help='write sentences to this file')
    parser.add_option('-x', '--train', dest='train', type='str', default=None,
                      help='train a new model using this labeled data file')
    parser.add_option('-c', '--svm', dest='svm', default=False,
                      action='store_true', help='use SVM instead of Naive Bayes for training')
    (options, args) = parser.parse_args()

    ## get test file
    if len(args) > 0:
        options.test = args[0]
        if not os.path.isfile(options.test): sbd_util.die('test path [%s] does not exist' %options.test)
    else:
        options.test = None
        if not options.train: sbd_util.die('you did not specify either train or test!')

    ## create model path
    if not options.model_path.endswith('/'): options.model_path += '/'
    if options.train:
        if not os.path.isfile(options.train): sbd_util.die('model path [%s] does not exist' %options.train)
        if os.path.isdir(options.model_path): sbd_util.die('model path [%s] already exists' %options.model_path)
        else: os.mkdir(options.model_path)
    else:
        if not os.path.isdir(options.model_path):
            options.model_path = install_root + options.model_path
            if not os.path.isdir(options.model_path):
                sbd_util.die('model path [%s] does not exist' %options.model_path)

    ## create a model
    if options.train:
        model = build_model(options.train, options)

    if not options.test: sys.exit()

    print options.svm
    print options.test
    
    ## test
    if not options.train:
        if 'svm' in options.model_path: options.svm = True
        model = load_sbd_model(options.model_path, options.svm)
    if options.output: options.output = open(options.output, 'w')

    test = get_data(options.test, tokenize=True)
    test.featurize(model, verbose=True)
    model.classify(test, verbose=True)
    print options.tokenize
    print options.output
    test.segment(use_preds=True, tokenize=options.tokenize, output=options.output)


