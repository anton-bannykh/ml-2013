import numpy
from collections import Counter, namedtuple

accurancy = lambda res : (res.tp + res.tn) / (res.tp + res.tn + res.fp + res.fn)
precision = lambda res : 0 if res.tp + res.fp == 0 else res.tp / (res.tp + res.fp)
recall = lambda res : 0 if res.tp + res.fn == 0 else res.tp / (res.tp + res.fn)
f1 = lambda res : 0 if res.tp == 0 else 2 * precision(res) * recall(res) / (precision(res) + recall(res))

def checkClassifier(classifier, X, y):
    t = {(1, 1) : 'tp', (-1, -1) : 'tn', (1, -1) : 'fn', (-1, 1) : 'fp'}
    c = Counter()
    for (xi, yi) in zip(X, y):
        yip = classifier(xi)
        c[t[(yip, yi)]] += 1
    return namedtuple('Result', ['tp', 'tn', 'fp', 'fn'])(c['tp'], c['tn'], c['fp'], c['fn'])

def split(X, y, n):
    perm = numpy.random.permutation(y.size)
    s, t = perm[:n], perm[n:]
    return X[s], y[s], X[t], y[t]

def loadData(filename="wdbc.data", url="http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"):
    import os
    if not os.path.exists(filename):
        import urllib.request
        import shutil

        with urllib.request.urlopen(url) as fu, open(filename, 'wb') as f:
            shutil.copyfileobj(fu, f)

    f = open('wdbc.data')
    ss = f.readlines()
    X = numpy.array([list(map(float, s.split(',') [2:])) for s in ss])
    y = numpy.array([1 if s.split(',')[1] == 'M' else -1 for s in ss])
    return X, y
