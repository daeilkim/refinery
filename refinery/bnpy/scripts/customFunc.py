'''
customFunc.py
A custom function that we can use to hook into the BNPy analysis. Runs at every minibatch or lap depending on the
the type of analysis.

onLapComplete() run after every complete lap through all B batches
onBatchComplete() run after every complete visit (Mstep, Estep, Sstep, ELBOstep) to a single batch
onAlgorithmComplete() run after the algorithm converges/reaches maximum number of laps

'''
import redis
msgServer = redis.StrictRedis()

def onLapComplete(hModel, percentDone, customFuncArgs):
    update = str(percentDone) + "% Done"
    msgServer.publish('analysis', "%s" % (update))
    print "onLapComplete"

def onBatchComplete(hModel, percentDone, customFuncArgs):
    update = str(percentDone) + "% Done"
    msgServer.publish('analysis', "%s" % (update))
    print "onBatchComplete"

def onAlgorithmComplete(hModel, percentDone, customFuncArgs):
    msgServer.publish('analysis', "%s" % ('status:Analysis Finished'))
    print "onAlgorithmComplete"