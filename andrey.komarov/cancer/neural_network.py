from common import *
from scipy.optimize import minimize
import numpy.random as r
import numpy

r.seed(3)

def mkNNClassifier(X, y, k, eta=2, IT = 5):
    Xm = X.mean(axis=(0))
    X = (X - Xm) / Xm
    y = (y + 1) // 2
    h = lambda z : 1. / (1 + numpy.exp(-z))
    N, n = X.shape
    m = 1
    w1 = r.random([k, n + 1]) - 0.5
    w2 = r.random([m, k + 1]) - 0.5

    for i in range(IT):
        # print (i)
        for o, t in zip(X, y):
            oo = numpy.hstack([o, 1])
            o1 = h(w1.dot(oo))
            oo1 = numpy.hstack([o1, 1])
            o2 = h(w2.dot(oo1))
            delta2 = o2 * (1 - o2) * (o2 - t)
            delta1 = o1 * (1 - o1) * (delta2.dot(w2[:,:-1]))
            dw1 = delta1.reshape([-1,1]).dot(oo.reshape([1,-1]))
            dw2 = delta2.reshape([-1,1]).dot(oo1.reshape([1,-1]))
            w1 -= eta * dw1
            w2 -= eta * dw2

    def feed(o, w1, w2):
        oo = numpy.hstack([o, 1])
        o1 = h(w1.dot(oo))
        oo1 = numpy.hstack([o1, 1])
        o2 = h(w2.dot(oo1)) - 0.5
        return numpy.sign(o2[0])

    return lambda o : feed((o - Xm) / Xm, w1, w2)

X, y = loadData()

bestC, bestCF1, bestRes = None, None, None

XStudy, yStudy, XTest, yTest = split(X, y, int(0.5 * y.size))

bestK, bestCF1, bestRes = None, None, None

for k in range(5, 40, 5):
    XStudy, yStudy, XTest, yTest = split(X, y, int(0.5 * y.size))
    classifier = mkNNClassifier(XStudy, yStudy, k)
    res = checkClassifier(classifier, XTest, yTest)
    f1Here = f1(res)
    print ("Neural network with k=%d, F1=%f"%(k, f1Here))
    if not bestCF1 or f1Here > bestCF1:
        bestCF1, bestK, bestRes = f1Here, k, res

for (w, f) in [('accurancy', accurancy), ('precision', precision), ('recall', recall), ('F1 measure', f1)]:
    print ("Neural network with k=%d %s: %f"%(bestK, w, f(bestRes)))
