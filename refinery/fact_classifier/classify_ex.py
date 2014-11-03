from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
import pickle
v = DictVectorizer()

#TODO : need to tokenize the words before using them as features!

def main():

    def munge(s):
        ps = s.split()
        label = int(ps[0])
        ws = defaultdict(int)
        for w in ps[1:]:
            ws[w] += 1
        return [label,ws]

    data = [munge(l.strip()) for l in open("/home/chonger/Downloads/annotations.txt")]

    labels = [x[0] for x in data]
    dicts = [x[1] for x in data]

    feats = v.fit_transform(dicts)

    ttsplit = int(len(labels) * .8)
    clf = svm.SVC(kernel='linear', class_weight={1: 10})
    #clf = svm.SVC()
    clf.fit(feats[:ttsplit],labels[:ttsplit])

    print clf.score(feats[ttsplit:],labels[ttsplit:])

    tot = defaultdict(int)
    tr = defaultdict(int)
    for ex in labels[ttsplit:]:
        tr[ex] += 1

    for ex in feats[ttsplit:]:
        tot[(clf.predict(ex).tolist())[0]] += 1

    print tr
    print tot

    print feats[0]
    print feats[1]

    f = open("/home/chonger/factsvm",'w')
    pickle.dump(clf,f)
    f.close()

    f = open("/home/chonger/factfeat",'w')
    pickle.dump(v,f)
    f.close()


if __name__ == "__main__":
    main()
