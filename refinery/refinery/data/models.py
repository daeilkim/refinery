# models.py contains code for defining the user object and behavior which will be used throughout the site
from refinery import db, app
import datetime
from refinery.webapp.pubsub import msgServer
from collections import defaultdict
import random,os,re,codecs


from collections import defaultdict
import pickle


# Defines a User class that takes the database and returns a User object that contains id,nickname,email
class User(db.Model):
    
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(64), index = True, unique = True)
    password = db.Column(db.String(64), index = True)
    email = db.Column(db.String(120), index = True, unique = True)
    image = db.Column(db.String(100))

    #datasets = db.relationship('Dataset', backref = 'author', lazy = 'dynamic')
    
    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email
        self.image = None

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return unicode(self.id)

    def __repr__(self):
        return '<User %r>' % (self.username)

    def check_password(self, proposed_password):
        if self.password != proposed_password:
            return False
        else:
            return True

class Experiment(db.Model):

    id = db.Column(db.Integer, primary_key = True)
    extype = db.Column(db.String(100)) # name of the model i.e topic_model
    status = db.Column(db.Text) # status (idle,start,inprogress,finish)

    
    def __init__(self, owner_id, extype):
        self.owner_id = owner_id
        self.extype = extype
        self.status = 'idle'
        
    def getExInfo(self):
        if(self.extype == "topicmodel"):
            return TopicModelEx.query.filter_by(ex_id=self.id).first()
        elif(self.extype == "summarize"):
            return SummarizeEx.query.filter_by(ex_id=self.id).first()
        else:
            return None


class TopicModelEx(db.Model):

    id = db.Column(db.Integer, primary_key = True)
    ex_id = db.Column(db.Integer, db.ForeignKey('experiment.id'))
    viz_data = db.Column(db.PickleType) #the top words and topic proportions
    nTopics = db.Column(db.Integer)
    stopwords = db.Column(db.PickleType) 
    
    def __init__(self, ex_id, nTopics):
        self.ex_id = ex_id
        self.viz_data = None
        self.nTopics = nTopics
        self.stopwords = []

class SummarizeEx(db.Model):

    id = db.Column(db.Integer, primary_key = True)
    ex_id = db.Column(db.Integer, db.ForeignKey('experiment.id'))
    current_summary = db.Column(db.PickleType) # a list of sentences in the current summary
    top_candidates = db.Column(db.PickleType) # a list of top ranked candidate sentences
    sents = db.Column(db.PickleType) 
    running = db.Column(db.Integer)
    
    def __init__(self, ex_id):
        self.ex_id = ex_id
        self.current_summary = []
        self.top_candidates = []
        self.running = 0


class DataDoc(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    data_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    doc_id = db.Column(db.Integer, db.ForeignKey('document.id'))

    def __init__(self, dset, doc):
        self.data_id = dset
        self.doc_id = doc

class Document(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(256)) #the name of this file
    path = db.Column(db.String(256)) #the path to the raw file data
    
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.sents = []

    def getStaticURL(self):
        print "!!!!!!!","/" + os.path.relpath(self.path,"refinery")
        return "/" + os.path.relpath(self.path,"refinery")

    def getText(self):
        lines = [line for line in codecs.open(self.path,"r","utf-8")]
        return "\n".join(lines)

def tokenize_sentence(text):
    ''' Returns list of words found in String. Matches A-Za-z and \'s '''
    wordPattern = "[A-Za-z]+[']*[A-Za-z]*"  
    wordlist = re.findall( wordPattern, text)
    return wordlist    


class Folder(db.Model):

    id = db.Column(db.Integer, primary_key = True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id')) # the dataset that was used
    docIDs = db.Column(db.PickleType)  #hopefully, this can be a dictionary of included docIDs
    name = db.Column(db.String)

    tm_id = db.Column(db.Integer, db.ForeignKey('experiment.id')) 
    sum_id = db.Column(db.Integer, db.ForeignKey('experiment.id'))

    vocabSize = db.Column(db.Integer)

    dirty = db.Column(db.String(20))
        
    def __init__(self, dataset_id, name, docIDs):
        self.dataset_id = dataset_id
        self.docIDs = docIDs
        self.name = name
        self.tm_id = None
        self.sum_id = None
        self.dirty = "dirty"

    def numSents(self):
        s = Experiment.query.get(self.sum_id).getExInfo()
        if s.sents:
            return sum([len(s.sents[ss]) for ss in s.sents])
        return 0

    def numTopics(self):
        tm = Experiment.query.get(self.tm_id)
        return tm.getExInfo().nTopics

    def topicModelEx(self):
        return Experiment.query.get(self.tm_id)

    def sumModelEx(self):
        return Experiment.query.get(self.sum_id)

    def initialize(self):

        ex1 = Experiment(self.id, "topicmodel")
        db.session.add(ex1)
        db.session.commit()
        tm = TopicModelEx(ex1.id,10)
        db.session.add(tm)
        db.session.commit()
        self.tm_id = ex1.id

        ex2 = Experiment(self.id, "summarize")
        db.session.add(ex2)
        db.session.commit()
        ei = SummarizeEx(ex2.id)
        db.session.add(ei)
        db.session.commit()
        self.sum_id = ex2.id

        db.session.commit()

    def documents(self):  # a generator for documents
        dataset = Dataset.query.get(self.dataset_id)
        for d in dataset.documents:
            if d.id in self.docIDs:
                yield d

    def N(self):
        dataset = Dataset.query.get(self.dataset_id)
        tot = len(list(self.documents()))
        return tot

    def all_docs(self):
        return sorted([Document.query.get(x.doc_id) for x in self.documents()],key=lambda x: x.id)

        
    def preprocTM(self, username, min_doc, max_doc_percent):

        #we need to add options, like to get rid of xml tags!
        
        STOPWORDFILEPATH = 'refinery/static/assets/misc/stopwords.txt'
        stopwords = set([x.strip() for x in open(STOPWORDFILEPATH)])
        
        allD = self.all_docs()
        
        nDocs = len(allD)
        
        WC = defaultdict(int)
        DWC = defaultdict( lambda: defaultdict(int) )

        def addWord(f,w):
            WC[w] += 1
            DWC[f][w] += 1

        c = 0.0
        prev = 0
        for d in allD:
            filE = d.path
            
            c += 1.0
            pc = int(c / float(nDocs) * 100)
            if pc > prev:
                prev = pc
                s = 'pprog,Step 1,' + str(self.id) + "," + str(pc)
                msgServer.publish(username + 'Xmenus', "%s" % s)
            
            [[addWord(filE,word) for word in tokenize_sentence(line) if word.lower() not in stopwords] for line in open(filE)] 

        # now remove words with bad appearace stats
        to_remove = []
        c = 0.0
        oldpc = -1
        for w in WC:
            c += 1.0
            pc = int(c/float(len(WC)) * 100)
            if not oldpc == pc:
                s = 'pprog,Step 2,' + str(self.id) + "," + str(pc)
                #print s
                msgServer.publish(username + 'Xmenus', "%s" % s)
                oldpc = pc
            has_w = [d for d,m in DWC.items() if w in m]
            n_has_w = len(has_w)
            doc_percent = float(n_has_w)/float(nDocs)
            #print w,doc_percent,n_has_w
            if n_has_w < min_doc or doc_percent > max_doc_percent:
                [DWC[d].pop(w,None) for d in has_w]
                to_remove.append(w)
        [WC.pop(w,None) for w in to_remove]

        vocab = [w for w in WC]

        print "N VOCAB",len(vocab)
        
        v_enum = defaultdict(int)
        for w in vocab:
            v_enum[w] = len(v_enum) 
        d_enum = defaultdict(int)
        for f in allD:
            d_enum[f.path] = len(d_enum)
    
        outfile = open(self.wordcount_path(),'w')
        for d in allD:
            f = d.path
            m = DWC[f]
            fID = d_enum[f]
            for w, c in m.items():
                wID = v_enum[w]
                outfile.write(str(fID) + ',' + str(wID) + ',' + str(c) + '\n')
        outfile.close()

        self.vocabSize = len(vocab)
  
        outfile = open(self.vocab_path(),'w')
        [outfile.write(x + "\n") for x in vocab]
        outfile.close()


        self.dirty = "clean"
        db.session.commit()
                
    def preproc_path(self):
        dataset = Dataset.query.get(self.dataset_id)
        return "refinery/static/users/" + User.query.get(dataset.owner_id).username + "/processed/"

    def wordcount_path(self):
        return self.preproc_path() + str(self.id) + "_word_count.txt"

    def vocab_path(self):
        return self.preproc_path() + str(self.id) + "_vocab.txt"
    
    def unigram(self):
        wcfile = self.wordcount_path()
        lines = [x.strip().split(",") for x in open(wcfile,'r')]
        unigram_dist = [0.0 for _ in xrange(self.vocabSize)]
        for l in lines:
            wID = int(l[1])
            wC = int(l[2])
            unigram_dist[wID] += wC
        tot = sum(unigram_dist)
        return [x / tot for x in unigram_dist]
        #return unigram_dist

    def get_vocab_list(self):
        vocabfile = self.vocab_path()
        return [x.strip() for x in open(vocabfile,'r')]

class Dataset(db.Model):
    
    id = db.Column(db.Integer, primary_key = True)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String(100)) # name of the dataset
    summary = db.Column(db.Text) # summary of the dataset (optional)
    img = db.Column(db.String(100)) # path to dataset img

    owner = db.relationship('User', backref = 'datasets')
    folders = db.relationship('Folder', backref = 'dataset', lazy = 'dynamic')
    documents = db.relationship('DataDoc', backref = 'docdataset', lazy = 'dynamic')

    def get_folders(self):
        return self.folders.order_by(Folder.id)

    def __init__(self, owner, name, summary, img=None):
        self.owner_id = owner
        self.name = name
        self.summary = summary
        if img is None:
            random_img = random.choice(os.listdir(app.config['RANDOM_IMG_DIRECTORY']))
            self.img = os.path.join("assets/images/random", random_img)
        else:
            self.img = img
        


        




