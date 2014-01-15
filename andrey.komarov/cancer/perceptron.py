from common import *

def mkPerceptronClassifier(preX, y):
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

X, y = loadData()
XStudy, yStudy, XTest, yTest = split(X, y, int(0.8 * y.size))
perceptron = mkPerceptronClassifier(XStudy, yStudy)
res = checkClassifier(perceptron, XTest, yTest)
for (w, f) in [('accurancy', accurancy), ('precision', precision), ('recall', recall), ('F1 measure', f1)]:
    print ("Perceptron %s: %f"%(w, f(res)))
