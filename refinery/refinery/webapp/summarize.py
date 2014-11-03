'''

Handles the summarization functionality of Refinery

'''

import os
from refinery import app, db
from refinery.webapp.pubsub import msgServer
from refinery.data.models import Experiment, Folder, tokenize_sentence, Document
from flask import request, render_template, jsonify
from refinery import celery
from math import log
from collections import defaultdict
import math
import lib.sbd as sbd
import json
import codecs

NUM_RESULTS = 200

def pubsub_name(username, experiment_id):
    '''
    constructs the pubsub for summarization
    for now this is not used, but could be if pubsub was added
    '''
    return username + '_summary_' + str(experiment_id)

@app.route('/viz_sum/<int:folder_id>/')
def summarize_viz(folder_id=None):
    '''serves the summarization page'''
    [folder, sum_ex, ex_info] = get_data(folder_id)
    data = json.dumps(ex_info.current_summary)
    return render_template("summarize.html", folder=folder,
                           ex=sum_ex, ei=ex_info, data=data)

def set_sum_status(username, folder_id, sum_ex, status):
    '''set summarization status for a folder and publish to the main menu'''
    sum_ex.status = status
    channel = username + "Xmenus"
    msgServer.publish(channel, "sumstatus," + str(folder_id) + "," + status)

def get_data(folder_id):
    '''Grabs the relevant summarize DB items for a folder ID'''
    folder = Folder.query.get(folder_id)
    sum_ex = Experiment.query.get(folder.sum_id)
    ex_info = sum_ex.getExInfo()
    return [folder, sum_ex, ex_info]

@celery.task()
def learn_summarize_model(username, folder_id):
    '''
    The Learning step for the summarize model.  Each document in the folder is
    sentence segmented, and each sentence classified as being a fact or not.
    '''

    #SETUP

    STOPWORDFILEPATH = 'refinery/static/assets/misc/stopwords.txt'
    [folder, sum_ex, ex_info] = get_data(folder_id)
    set_sum_status(username, folder_id, sum_ex, 'inprogress')
    db.session.commit()


    #map vocab words to their index
    vocab = {}
    for word in open(folder.vocab_path()):
        vocab[word.strip()] = len(vocab)

    stopwords = set([x.strip() for x in open(STOPWORDFILEPATH)])

    #sbd setup
    sbd_model_path = os.path.abspath("") + '/lib/model_svm/'
    sbd_model = sbd.load_sbd_model(sbd_model_path, True)
    '''
    #fact svm setup
    fid = open("fact_classifier/factsvm")
    fact_svm = pickle.load(fid)
    fid.close()
    fid = open("fact_classifier/factfeat")
    feat_extractor = pickle.load(fid)
    fid.close()
    '''
    # START WORKING

    all_documents = folder.all_docs()
    allsents = dict()
    total_docs = float(len(all_documents))
    last_prog = 0
    count = 0.0

    for doc in all_documents:

        #send progress info
        count += 1.0
        update = int(count / float(total_docs) * 100)
        if update != last_prog:
            last_prog = update
            msg = 'sum_prog,' + str(folder_id) + "," + str(update)
            msgServer.publish(username + "Xmenus", msg)

        #get the raw file text
        filE = doc.path
        raw_text = ""
        fid = codecs.open(filE, "r", "utf-8")
        for line in fid:
            tline = line.strip()
            raw_text += " " + tline
        fid.close()

        #sentence boundary detection
        sents = sbd.sbd_text(sbd_model, raw_text, do_tok=False)

        def filter_sentences():
            ''' this generator uses fact classification to filter sentences'''
            for sent in sents:
                if len(sent) > 200:
                    continue
                words = tokenize_sentence(sent)
                yield [sent, words]

                '''
                #actual classifier, commented out for now...
                ws = defaultdict(int)
                for w in words:
                    ws[w] += 1
                pred = fact_svm.predict(feat_ex.transform(ws))
                if(pred == 1):
                    yield [sent,words]
                '''


        def get_sentences():
            '''tokenize and drop stopwords to get word count dicts'''
            for sent, words in filter_sentences():
                word_counts = defaultdict(int)
                good_words = [word for word in words if word in vocab and
                              word.lower() not in stopwords]
                for word in good_words:
                    word_counts[vocab[word]] += 1
                if len(word_counts) > 0:
                    yield [sent, word_counts]

        allsents[doc.id] = [x for x in get_sentences()]

    #cleanup
    ex_info.sents = allsents
    set_sum_status(username, folder_id, sum_ex, 'finish')
    db.session.commit()

#TODO : this probably duplicates the code the provides browsing
@app.route('/get_doc_fulltext/', methods=["POST"])
def get_doc_fulltext():
    '''returns the full text of the document whose ID was posted'''
    didx = int(request.form['didx'])
    doc = Document.query.get(didx)
    path = doc.path
    lines = [l.strip() for l in open(path)]
    text = "\n".join([l for l in lines if len(l) > 0])
    return jsonify(fulltext=text,)


@app.route('/sum_add/<int:folder_id>/', methods=["POST"])
def sum_add(folder_id=None):
    '''
    Add the POSTed sentences to the current summary
    '''
    [_, _, ex_info] = get_data(folder_id)

    sents = json.loads(request.form['sents'])

    cursum = [x for x in ex_info.current_summary]
    cursum.extend(sents)
    ex_info.current_summary = cursum
    db.session.commit()

    return jsonify(data=ex_info.current_summary)


def get_top_sentences(metric, folder_id, query_sents):
    '''
    using a similarity metric and a query, get the top candidates from a folder
    '''

    [_, _, ex_info] = get_data(folder_id)

    used_sents = dict()
    for _, _, doc_index, sent_index in ex_info.current_summary:
        used_sents[(doc_index, sent_index)] = 1

    current_counts = defaultdict(int)
    current_total = 0.0
    for _, word_counts, _, _ in query_sents:
        for word_index, word_count in word_counts.items():
            current_counts[int(word_index)] += word_count
            current_total += word_count

    def sentiter():
        '''yield the sentences that have not been used yet'''
        for doc_index, sents in ex_info.sents.items():
            sent_index = 0
            for sent, word_counts in sents:
                if (doc_index, sent_index) not in used_sents:
                    yield [sent, word_counts, doc_index, sent_index]
                sent_index += 1

    scored = [[x, metric(x[1], current_counts, current_total)] for x in sentiter()]

    results = [data for data, _
               in sorted(scored, key=lambda x: x[1])[:NUM_RESULTS]]

    return jsonify(results=results)

@app.route('/get_variety/<int:folder_id>/', methods=["POST"])
def get_variety(folder_id=None):
    '''
    post json of doc index / sentence index pairs to form a query

    get the top NUM_RESULTS new sentences by cosine dist between the sentences
    BOW multinomial and the BOW multinomial for the query sentences

    the posted sentences have the form [text,word_count,doc_index,sent_index]
    '''
    sents = json.loads(request.form['sents'])

    folder = Folder.query.get(folder_id)
    folder_unigram = folder.unigram()

    def kl_div(word_counts, current_counts, current_total):
        '''KL divergence'''
        
        new_total = sum([count for index, count in word_counts.items()])

        def get_distribution():
            '''get the nonzero elements of the candidate BOW dist'''
            for index, count in word_counts.items():
                prob = float(count)
                if index in current_counts:
                    prob += float(current_counts[index])
                prob /= (new_total + current_total)
                yield [index, prob]

            for index, count in current_counts.items():
                if index not in word_counts:
                    prob = float(count)
                    prob /= (new_total + current_total)
                    yield [index, prob]

        return sum([log(prob/folder_unigram[index])*prob
                    for index, prob in get_distribution()])

    return get_top_sentences(kl_div, folder_id, sents)

@app.route('/get_similar/<int:folder_id>/', methods=["POST"])
def get_similar(folder_id=None):
    '''
    post json of doc index / sentence index pairs to form a query

    get the top NUM_RESULTS new sentences by cosine sim between the sentence`s
    BOW multinomial and the BOW multinomial for the query sentences`
    '''

    sents = json.loads(request.form['sents'])

    current_length = 0.0
    for _, word_counts, _, _ in sents:
        for _, word_count in word_counts.items():
            current_length += math.pow(word_count, 2.0)
    current_length = math.sqrt(current_length)

    
    def cos_sim(word_counts, current_counts, _):
        """negative cosine similarity between a sentence and the query's BOW vector"""

        vector_length = math.sqrt(sum([math.pow(count, 2.0) for
                                       index, count in word_counts.items()]))
        dotprod = 0.0
        for index, count in word_counts.items():
            if index in current_counts:
                dotprod += count * current_counts[index]

        sim_val = dotprod / (vector_length * current_length)
        return -sim_val

    return get_top_sentences(cos_sim, folder_id, sents)

@app.route('/sum_delete/<int:folder_id>/', methods=["POST"])
def sum_delete(folder_id=None):
    '''
    posting json of doc index / sentence index pairs deletes them from
    the current summary, and returns the new summary to update the summary page
    '''

    [_, _, ex_info] = get_data(folder_id)

    sents = json.loads(request.form['sents'])

    indexmap = dict()
    for _, _, doc_index, sent_index in sents:
        index = str(doc_index) + "_" + str(sent_index)
        indexmap[index] = 0

    def filterer():
        '''
        only return the sentences that arent deleted
        '''
        for sent in ex_info.current_summary:
            _, _, doc_index, sent_index = sent
            index = str(doc_index) + "_" + str(sent_index)
            if index not in indexmap:
                yield sent

    ex_info.current_summary = [x for x in filterer()]
    db.session.commit()

    return jsonify(data=ex_info.current_summary)
