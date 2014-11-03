'''
BarsViz.py

Visualization tools for toy bars data for topic models.
'''
from matplotlib import pylab
import numpy as np

imshowArgs = dict(interpolation='nearest', cmap='bone', 
                  vmin=0.0, vmax=0.3)


def plotExampleBarsDocs(Data, docIDsToPlot=None,
                              vmax=None, nDocToPlot=9, doShowNow=True):
    pylab.figure()
    V = Data.vocab_size
    sqrtV = int(np.sqrt(V))
    assert np.allclose(sqrtV * sqrtV, V)
    if docIDsToPlot is not None:
      nDocToPlot = len(docIDsToPlot)
    else:
      docIDsToPlot = np.random.choice(Data.nDoc, size=nDocToPlot, replace=False)
    nRows = np.floor(np.sqrt(nDocToPlot))
    nCols = np.ceil(nDocToPlot / nRows)
    if vmax is None:
      DocWordArr = Data.to_sparse_docword_matrix().toarray()
      vmax = int(np.max(np.percentile(DocWordArr, 98, axis=0)))
      
    for plotPos, docID in enumerate(docIDsToPlot):
        # Parse current document
        start,stop = Data.doc_range[docID,:]
        wIDs = Data.word_id[start:stop]
        wCts = Data.word_count[start:stop]
        docWordHist = np.zeros(V)
        docWordHist[wIDs] = wCts
        squareIm = np.reshape(docWordHist, (np.sqrt(V), np.sqrt(V)))

        pylab.subplot(nRows, nCols, plotPos)
        pylab.imshow(squareIm, interpolation='nearest', vmin=0, vmax=vmax)
    if doShowNow:
      pylab.show()

def plotBarsFromHModel(hmodel, Data=None, doShowNow=True, figH=None,
                       compsToHighlight=None, sortBySize=False,
                       width=12, height=3, Ktop=None):
    if Data is None:
        width = width/2
    if figH is None:
      figH = pylab.figure(figsize=(width,height))
    else:
      pylab.axes(figH)
    K = hmodel.allocModel.K
    VocabSize = hmodel.obsModel.comp[0].lamvec.size
    learned_tw = np.zeros( (K, VocabSize) )
    for k in xrange(K):
        lamvec = hmodel.obsModel.comp[k].lamvec 
        learned_tw[k,:] = lamvec / lamvec.sum()
    if sortBySize:
        sortIDs = np.argsort(hmodel.allocModel.Ebeta[:-1])[::-1]
        sortIDs = sortIDs[:Ktop]
        learned_tw = learned_tw[sortIDs] 
    if Data is not None and hasattr(Data, "true_tw"):
        # Plot the true parameters and learned parameters
        pylab.subplot(121)
        pylab.imshow(Data.true_tw, **imshowArgs)
        pylab.colorbar()
        pylab.title('True Topic x Word')
        pylab.subplot(122)
        pylab.imshow(learned_tw,  **imshowArgs)
        pylab.title('Learned Topic x Word')
    else:
        # Plot just the learned parameters
        aspectR = learned_tw.shape[1]/learned_tw.shape[0]
        while imshowArgs['vmax'] > 2 * np.percentile(learned_tw, 97):
          imshowArgs['vmax'] /= 5
        pylab.imshow(learned_tw, aspect=aspectR, **imshowArgs)

    if compsToHighlight is not None:
        ks = np.asarray(compsToHighlight)
        if ks.ndim == 0:
          ks = np.asarray([ks])
        pylab.yticks( ks, ['**** %d' % (k) for k in ks])
    if doShowNow and figH is None:
      pylab.show()
