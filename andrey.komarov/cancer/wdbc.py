import numpy
import numpy.linalg as lin
import pylab
import matplotlib.pyplot as plt
from sklearn import svm
from collections import Counter

numpy.random.seed(1234)

def mkPCA(a, ndim=2):
    assert a.ndim == 2
    assert 1 <= ndim <= a.shape[1]
    a = a - numpy.mean(a, axis=0)
    eigenvalues,eigenvectors = lin.eig(numpy.dot(a.T, a))
    e = eigenvectors[numpy.argsort(eigenvalues)[:-ndim-1:-1]]
    return lambda v : numpy.dot(e, v.T)

def perceptronClassifier(preX, y):
    assert len(preX) == len(y)
    (n, d) = preX.shape
    X = numpy.column_stack((preX, numpy.ones(n)))
    w = numpy.zeros(d + 1)
    bestW, bestCnt = w, n
    for i in range(1000):
        for (x, yi) in zip(X, y):
            if numpy.sign(numpy.dot(w.T, x)) != yi:
                w += yi * x
        cnt = numpy.count_nonzero([numpy.sign(numpy.dot(w.T, x)) - yi for (x, yi) in zip(X, y)])
        if cnt < bestCnt:
            bestW, bestCnt = w.copy(), cnt
    return lambda v : numpy.sign(numpy.dot(numpy.hstack((v, 1)).T, bestW))

def split(X, y, n):
    perm = numpy.random.permutation(y.size)
    s, t = perm[:n], perm[n:]
    return X[s], y[s], X[t], y[t]

f = open('wdbc.data')
ss = f.readlines()

X = numpy.array([list(map(float, s.split(',') [2:])) for s in ss])
y = numpy.array([1 if s.split(',')[1] == 'M' else -1 for s in ss])
XStudy, yStudy, XTest, yTest = split(X, y, int(0.8 * y.size))

t = {(1, 1) : 'tp', (-1, -1) : 'tn', (1, -1) : 'fn', (-1, 1) : 'fp'}
c = Counter()

bestC, bestCPrec = None, None

for C in 2. ** numpy.arange(-5, 20):
    XStudy, yStudy, XTest, yTest = split(X, y, int(0.5 * y.size))
    classifier = svm.LinearSVC(C = C)
    classifier.fit(XStudy, yStudy)
    y1 = classifier.predict(XTest)
    c = Counter()
    for (yp, y2) in zip(y1, yTest):
        c[t[(yp, y2)]] += 1
    precision = 100 * (c['tp'] + c['tn']) / (sum(c.values()))
    if bestCPrec == None or precision > bestCPrec:
        bestC, bestCPrec = C, precision
    print ("SVM(C=%f) precision: %.3f%%"%(C, precision))
    print ("SVM(C=%f) recall:    %.3f%%"%(C, 100 * c['tp'] / (c['tp'] + c['fn'])))

print ("best C = %f with precision = %f"%(bestC, bestCPrec))


classifier = perceptronClassifier(XStudy, yStudy)

for (x, y1) in zip(XTest, yTest):
    y2 = classifier(x)
    c[t[(y1, y2)]] += 1

print ("################################")
precision = 100 * (c['tp'] + c['tn']) / sum(c.values())
print ("perceptron precision: %.3f%%"%(precision))
print ("perceptron recall:    %.3f%%"%(100 * c['tp'] / (c['tp'] + c['fn'])))


XStudy, yStudy, XTest, yTest = split(X, y, int(0.5 * y.size))
classifier =  svm.LinearSVC()
classifier.fit(XStudy, yStudy)

#a = numpy.array(X)
#f = mkPCA(a, 3)
#
#xyM = []
#xyB = []
#
#for s in open('wdbc.data').readlines():
#    a = s.split(',')
#    v = numpy.array(list(map(float, a[2:])))
#    (xyM if a[1] == 'M' else xyB).append(f(v))
#
#plt.plot([xy[0] for xy in xyM], [xy[1] for xy in xyM], 'ro')
#plt.plot([xy[0] for xy in xyB], [xy[1] for xy in xyB], 'b+')
#plt.show()