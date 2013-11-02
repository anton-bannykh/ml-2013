import numpy
import numpy.linalg as lin
import pylab
import matplotlib.pyplot as plt

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

classifier = perceptronClassifier(XStudy, yStudy)

ok = 0
all = 0
for (x, y) in zip(XTest, yTest):
    ok += 1 if y == classifier(x) else 0
    all += 1

print (ok, "of", all, " (%f%%) classified correctly"%(100 * ok / all))

#a = numpy.array(X)
#print(a.shape)
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