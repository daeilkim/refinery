import os, sys
from refinery import app,db
from pubsub import msgServer
from refinery.data.models import Dataset, Experiment, TopicModelEx, DataDoc, Folder
from flask import g, request, render_template, Response, jsonify

bnpydir = "bnpy/bnpy-dev"
sys.path.append(bnpydir)
import bnpy
from bnpy.data import WordsData
from customFunc import getModelState
from refinery import celery
import json
import numpy as np
import scipy as sp

from math import log
vocab = {}

def pubsub_name(d,e):
    return 'analysis_' + str(d) + "_" + str(e) 

def set_tm_status(username,fID,ex,st):
    '''set the topic model status and publish it to the menus pubsub'''
    ex.status = st
    msgServer.publish(username + "Xmenus","tmstatus," + str(fID) + "," + st)

@app.route('/<username>/viz_tm/<int:folder_id>/')
def topic_model_viz(username=None, folder_id=None):
    '''serve the topic model visualization'''
    f = Folder.query.get(folder_id)
    ex = Experiment.query.get(f.tm_id)
    ei = ex.getExInfo()
    vocabfile = f.vocab_path()
    vocab = [x.strip() for x in open(vocabfile,'r')]
    totalDox = f.N()
    return render_template("topicmodel.html", d=f, ex=ex, vocab=vocab, totalDox=totalDox, stopwords=" ".join(ei.stopwords))

@app.route('/<username>/data/<int:data_id>/<int:ex_id>/load_analysis_tm')
def load_analysis_tm(username=None, data_id=None, ex_id=None):
    '''
    load the topic modeling analysis

    this is called immediately by the topicmodeling viz page
    '''
    [topW,topic_probs,doc_tops,lms] = Experiment.query.get(ex_id).getExInfo().viz_data
    return jsonify(topW=topW,topic_probs=[x for x in topic_probs],doc_tops=doc_tops)

@app.route('/<username>/change_stopwords/<int:folder_id>', methods=['POST'])
def change_stopwords(username=None,folder_id=None):
    stopwords = request.form['stopwords'].strip().split()
    f = Folder.query.get(folder_id)
    ei = Experiment.query.get(f.tm_id).getExInfo()
    ei.stopwords = stopwords
    db.session.commit()
    return Response("200")

#TODO Duplicates code in summarize.py
@app.route('/<username>/get_doc_text', methods=['POST'])
def get_doc_text(username=None):
    
    filename = 'refinery/static/users/' + username + "/documents/" + request.form['filename']
    lines = [l.strip() for l in open(filename)]
    doctext = "\n".join([l for l in lines if len(l) > 0])
    
    return render_template("docview.html",doctext=doctext.decode('utf-8'))

@app.route('/<username>/set_num_topics/<int:folder_id>', methods=['POST'])
def set_num_topics(username=None, folder_id=None):
    '''set the number of topics for a folder'''
    v = request.form['v']
    f = Folder.query.get(folder_id)
    ex = Experiment.query.get(f.tm_id)
    tm = ex.getExInfo()
    tm.nTopics = int(v)
    set_tm_status(username,folder_id, ex, 'idle')
    db.session.commit()
    return Response(status='200')

@app.route('/<username>/data/<int:data_id>/<int:ex_id>/start_analysis_tm')
def run_analysis_tm(username=None, data_id=None, ex_id=None):
    '''Run topic modeling learning - passes it off to celery and returns'''
    msgServer.publish(pubsub_name(data_id,ex_id), "%s" % ('Starting analysis'))    
    run_topic_modeling.apply_async([username,data_id,ex_id])
    return Response(status='200')

# Run the analysis
@app.route('/<username>/make_subset/<int:folder_id>/<int:ex_id>',methods=['POST'])
def make_subset(username=None, folder_id=None, ex_id=None):

    dOld = Folder.query.get(folder_id)

    [topW,topic_probs,doc_tops,lms] = Experiment.query.get(ex_id).getExInfo().viz_data

    nTops = len(doc_tops[0])

    nDox = int(request.form['nDox'])
    searchwords = request.form['searchwords'].split(" ")
    topicblocks = [int(x) for x in request.form.getlist('blocks[]')]


    #avg each word's renormalized p(w|z)
    blockDist = [0.0 for _ in xrange(nTops)]

    tot = 0.0
    for v in topicblocks:
        if(v >= 0):
            blockDist[v] += 1.0
            tot += 1.0

    if(tot > 0):
        blockDist = [x/tot for x in blockDist]
    else:
        blockDist = [0.0 for x in blockDist]


    '''
    get target KL distribution from search terms
    sort by kl - below
    '''

    vocabfile = dOld.vocab_path()

    vocab = {}
    idx = 0
    vv = [x.strip() for x in open(vocabfile,'r')]
    for v in vv:
        vocab[v] = idx
        idx += 1

    searchSmooth = .0000001
    searchDist = [searchSmooth for _ in xrange(nTops)]


    sdTotal = searchSmooth * nTops
    for sw in searchwords:
        if sw in vocab:
            idx = vocab[sw]
            for tidx in xrange(nTops):
                for i,p in topW[tidx]:
                    if i == idx:
                        searchDist[tidx] += p
                        sdTotal += p
                        
    searchDist = [x/sdTotal for x  in searchDist]

    if len(request.form['searchwords'].strip()) > 0:
        targDist = [(a + b)/2.0 for a,b in zip(searchDist,blockDist)] # needs the search component
    else:
        targDist = blockDist


    targTotal = sum(targDist)
    if targTotal == 0:
        targDist = [1.0/float(nTops) for x in xrange(nTops)]
    else:
        targDist = [x/targTotal for x in targDist]

    
    L = np.array(lms,dtype=np.float16)
    searchLM = np.log(np.transpose(np.array(targDist,dtype=np.float16).dot(L)))
    D = sp.sparse.dok_matrix((dOld.N(),dOld.vocabSize),dtype=np.float16)

    wordcounts = open(dOld.wordcount_path(),'r')
    totalW = np.zeros((dOld.N()))
    for line in wordcounts:
        [d,w,c] = [int(x) for x in line.strip().split(",")]
        D[d,w] = c
        totalW[d] += c
    wordcounts.close()

    D = D.asformat("csr")


    lmScores = D.dot(searchLM)
    lmScores = np.divide(lmScores,totalW)
    lmScores = np.divide(np.power(2.0,np.multiply(lmScores,-1.0)),totalW)


    inds = np.argsort(lmScores)[:nDox].tolist()   #get the top indices
    probs = [doc_tops[idx] for idx in inds]         #get their weights

    topDocs = zip(inds,probs)

    aD = [d for d in dOld.all_docs()]

    def makedata():
        for td in topDocs:
            d = aD[td[0]]
            yield [d.id,d.name,td[1]]
    
    dd = [x for x in makedata()]

    # return json response of doc entities, targDist, doc_tops (for top docs)
    return jsonify(docs=dd,targDist=targDist)

@app.route('/<username>/create_subset/<int:folder_id>/<int:ex_id>',methods=['POST'])
def create_subset(username=None, folder_id=None, ex_id=None):

    docIDs = dict()
    for x in request.form.getlist('docIdxs[]'):
        docIDs[int(x)] = 0

    #first make the new folder
    fOld = Folder.query.get(folder_id)
    name = request.form['name']
    user_id = g.user.id
    
    fNew = Folder(fOld.dataset_id,name,docIDs)
    db.session.add(fNew)
    db.session.commit()

    fNew.initialize()
    
    return Response(status='200')


@celery.task()
def run_topic_modeling(username,folder_id,ex_id):

    d = Folder.query.get(folder_id)
    ex = Experiment.query.get(ex_id)
    set_tm_status(username,folder_id, ex,'inprogress')
    db.session.commit()
    
    exinfo = ex.getExInfo()
        
    # CREATE WORD DATA
    datafile = d.wordcount_path()
    vocabfile = d.vocab_path()

    vocab = {}
    idx = 0
    vv = [x.strip() for x in open(vocabfile,'r')]
    for v in vv:
        vocab[idx] = v
        idx += 1
        
    lines = [x.strip().split(",") for x in open(datafile,'r')]
        
    docrange = []
    word_id = []
    word_count = []
    start = 0
    cur = -1
    curD = -1
    for l in lines:
        cur += 1
        word_id.append(int(l[1]))
        word_count.append(int(l[2]))

        dID = int(l[0])
        if(curD == -1):
            curD = dID
        if(curD != dID):
            docrange.append([start,cur-1])
            start = cur-1
            curD = dID
    docrange.append([start,cur+1])
    data = WordsData(word_id,word_count,docrange,len(vocab),vocab,len(docrange))

    # RUN Topic Modeling in BNPY

    a = {"tm_id":str(d.id), "username":username}

    hmodel = bnpy.Run.run(data, 'HDPModel', 'Mult', 'VB', doSaveToDisk=False, K=exinfo.nTopics,
                          nLap=100, initname="randomfromprior",
                          customFuncPath="refinery/webapp/", customFuncArgs=json.dumps(a))
    '''
                          moves='birth,merge', birthPerLap=10, \
                          mergePerLap=10, nFreshLap=25)
    '''

    
    exinfo.viz_data = getModelState(hmodel[0],hmodel[1],100)

    set_tm_status(username,folder_id, ex,'finish')

    db.session.commit()

