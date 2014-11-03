"""
views.py

Created by Dae Il Kim on 2013-03-25.

main_menu.py

Name changed by Ben Swanson on 2014-09-24

Copyright (c) 2014 Refinery. All rights reserved. Respect that shit or else.

Or else what?  Exactly.

"""

from flask import request, g, render_template, jsonify, Response, json
from flask.ext.login import current_user,login_required

import celery

from refinery import app, db
from refinery.data.models import Dataset, Experiment, Folder
from refinery.webapp.pubsub import msgServer

import refinery.webapp.admin as admin
import refinery.webapp.upload as upload
import refinery.webapp.summarize as summarize
import refinery.webapp.topicmodel as topicmodel

@app.before_request
def before_request():
    '''set global variable g representing current login'''
    g.user = current_user

@app.route('/main')
@login_required
def manage_data():
    '''Main Data page (home page)'''
    result = g.user.datasets
    if result:
        data = [[d, [f for f in d.get_folders()]] for d in result]
    else:
        data = []
    return render_template('data_list.html', data=data, nodata=(len(data) == 0))

@app.route('/delete_folder/<int:folder_id>')
@login_required
def delete_folder(folder_id=None):
    '''delete a folder'''
    folder = Folder.query.get(folder_id)
    tm_ex = Experiment.query.get(folder.tm_id)
    sum_ex = Experiment.query.get(folder.sum_id)
    tm_ei = tm_ex.getExInfo()
    sum_ei = sum_ex.getExInfo()
    db.session.delete(tm_ei)
    db.session.delete(sum_ei)
    db.session.delete(folder)
    db.session.commit()
    db.session.delete(tm_ex)
    db.session.delete(sum_ex)
    db.session.commit()
    return Response(status='200')

@app.route('/delete_dataset/<int:data_id>')
@login_required
def delete_dataset(data_id=None):
    '''delete a dataset'''
    dset = Dataset.query.get(data_id)
    for folder in dset.folders:
        delete_folder(folder.id)
    db.session.delete(dset)
    db.session.commit()
    return Response(status='200')


@app.route('/data/edit_dataset/<int:data_id>',
           methods=["GET", "POST"])
@login_required
def edit_dataset(data_id=None):
    '''Edit the title and summary of the dataset'''
    dset = Dataset.query.get(data_id)
    dset.name = request.form['name']
    dset.summary = request.form['sum']
    db.session.commit()
    return Response(status='200')


#TODO : These could be combined
@app.route('/<username>/start_tm/<int:folder_id>')
def start_tm(username=None, folder_id=None):
    '''start the topic model learning'''
    folder = Folder.query.get(folder_id)
    if folder.dirty == 'clean':
        ex = Experiment.query.get(folder.tm_id)
        if ex.status == 'idle':
            topicmodel.run_topic_modeling.apply_async([username,
                                                       folder_id, ex.id])
            return jsonify(command='stay')
        elif ex.status == 'inprogress':
            return jsonify(command='stay')
        else:
            return jsonify(command='go')
    else:
        return jsonify(command='fail')

@app.route('/<username>/start_sum/<int:folder_id>')
def start_sum(username=None, folder_id=None):
    '''start the summary model learning'''
    folder = Folder.query.get(folder_id)
    if folder.dirty == 'clean':
        ex = Experiment.query.get(folder.sum_id)
        if ex.status == 'idle':
            summarize.learn_summarize_model.apply_async([username,
                                                  folder_id])
            return jsonify(command='stay')
        elif ex.status == 'inprogress':
            return jsonify(command='stay')
        else:
            return jsonify(command='go')
    else:
        return jsonify(command='fail')

"""
@app.route('/viz/<int:ex_id>/')
def experiment_viz(ex_id=None):
    '''call this to switch experiment view'''
    ex = Experiment.query.get(ex_id)
    if ex.extype == "topicmodel":
        return topicmodel.topic_model_viz(ex)
    if ex.extype == "summarize":
        return summarize.summarize_viz(ex)
    return Response("500")
"""

@app.route('/docjson/<int:folder_id>')
def get_docs_json(folder_id=None):
    folder = Folder.query.get(folder_id)
    documents = [[doc.name,doc.id,doc.getText()] for doc in folder.all_docs()]
    return jsonify(documents=documents)

@app.route('/<username>/start_preproc/<int:folder_id>/')
def start_preproc(username=None, folder_id=None):
    '''start preprocessing'''
    folder = Folder.query.get(folder_id)
    if folder.dirty == "dirty":
        folder.dirty = "working"
        db.session.commit()
        preproc_dataset.apply_async([username, folder.id])
        return jsonify(data_status='start')
    else:
        return jsonify(dats_status=folder.dirty)

@celery.task()
def preproc_dataset(username, fid):
    '''preprocess the dataset'''
    
    #the username is required so that the correct pubsub can be used by preproc

    folder = Folder.query.get(fid)
    vocab_min_doc = 2  # minimum number of documents a word must be in
    max_percent = .8  # maximum percentage of documents a word can be in
    folder.preprocTM(username, vocab_min_doc, max_percent)
    pubsub_msg = 'proc,' + str(folder.dataset_id) + "," + str(fid) + ",clean"
    msgServer.publish(username + 'Xmenus', "%s" % pubsub_msg)

