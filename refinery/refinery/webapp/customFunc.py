'''
customFunc.py

A custom function that we can use to hook into the BNPY analysis.

onLapComplete() run after every complete lap through all B batches
onBatchComplete() run after every complete visit (Mstep, Estep, Sstep, ELBOstep) to a single batch
onAlgorithmComplete() run after the algorithm converges/reaches maximum number of laps

'''
import redis
import numpy as np
import json

msgServer = redis.StrictRedis(socket_timeout=20)
#from pubsub import msgServer

def getModelState(hmodel,LP,nTopW):

    num_topics = hmodel.allocModel.K 

    def lm_info():
        for k in xrange(num_topics):
            lamvec = hmodel.obsModel.comp[k].lamvec        #bag of words weights
            elamvec = lamvec / lamvec.sum()                #renormalized weights
            inds = np.argsort(elamvec)[-nTopW:].tolist()   #get the top indices
            inds.reverse()
            probs = [elamvec[idx] for idx in inds]         #get their weights
            yield [elamvec,zip(inds,probs)]

    topW = []
    lms = []
    for lm,tops in lm_info():
        topW.append(tops)
        lms.append(lm)

    topic_props = hmodel.allocModel.Ebeta

    def renormalize(vec):
        tot = sum(vec)
        return [x/tot for x in vec]

    doc_tops = [renormalize(x[:-1]) for x in LP['alphaPi']] #topic posteriors for each document, drop the last value because bnpy

    return [topW,topic_props,doc_tops,lms]

def onLapComplete(hmodel, percentDone, customFuncArgs):
    
    update = str(percentDone * 100)
    customArgs = json.loads(customFuncArgs)
    tm_id = customArgs["tm_id"]
    username = customArgs["username"]

    msgServer.publish(username + "Xmenus",'tm_prog,' + tm_id + "," + update)

'''

For now we dont use these hooks, but bnpy allows them

'''

def onBatchComplete(hModel, percentDone, customFuncArgs):
    print "onBatchComplete!"

def onAlgorithmComplete(hModel, percentDone, customFuncArgs):
    print "onAlgorithmComplete!"
