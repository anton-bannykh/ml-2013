import numpy as np

from common import *
from sys import stderr

# learning rate, hidden layer size
def train_neural_network(training_set, alpha, H):
    datax = [x.features for x in training_set]
    datay = [x.correct for x in training_set]

    POSITIVE = 1.0
    NEGATIVE = 0.0

    for i in range(len(datay)):
        if datay[i] == -1:
            datay[i] = NEGATIVE
        else:
            datay[i] = POSITIVE

    dim = len(datax[0])

    n = dim
    M = 1

    w1 = None
    w2 = None

    def logistic(x):
        return 1.0 / (1 + np.exp(-x))

    def logisticd(x):
        return logistic(x) * (1 - logistic(x))

    vlogistic = np.vectorize(logistic)
    vlogisticd = np.vectorize(logisticd)

    # w1 = np.zeros((n + 1, H))
    # w2 = np.zeros((H + 1, M))
    w1 = np.random.random((n + 1, H))
    w2 = np.random.random((H + 1, M))

    xx = None
    hsum = None
    h = None
    asum = None
    a = None

    def forward(x):
        nonlocal xx
        nonlocal hsum
        nonlocal h
        nonlocal asum
        nonlocal a
        xx = np.zeros(len(x) + 1)
        xx[0] = -1.0
        xx[1:] = x
        hsum = xx.dot(w1)
        h = np.zeros(1 + H)
        h[0] = -1.0
        h[1:] = vlogistic(hsum)
        asum = h.dot(w2)
        a = vlogistic(asum)
        return a

    def backward(y):
        nonlocal w1
        nonlocal w2
        ea = a - y
        dera = vlogisticd(asum)
        eh = np.multiply(ea, dera).dot(np.transpose(w2[1:][:]))
        derh = vlogisticd(hsum)
        dw2 = alpha * np.outer(h, np.multiply(ea, dera))
        dw1 = alpha * np.outer(xx, np.multiply(eh, derh))
        w2 -= dw2
        w1 -= dw1


    data = list(zip(datax, datay))
    for i in range(1000):
        x, y = random.choice(data)
        res = forward(x)
        backward(y)


    def classify(x):
        y = forward(x)
        d1 = np.linalg.norm(y - NEGATIVE)
        d2 = np.linalg.norm(y - POSITIVE)
        return -1 if d1 < d2 else 1

    return classify

#def train_neural_network(training_set, shape, lam):
#    datax = [x.features for x in training_set]
#    datay = [x.correct for x in training_set]
#
#    POSITIVE = np.array([0, 1])
#    NEGATIVE = np.array([1, 0])
#
#    for i in range(len(datay)):
#        if datay[i] == -1:
#            datay[i] = NEGATIVE
#        else:
#            datay[i] = POSITIVE
#
#    dim = len(datax[0])
#
#    shape = [dim] + shape + [2]
#
#    ww = [None for _ in range(len(shape) - 1)]
#    ss = [None for _ in shape]
#    aa = [None for _ in shape]
#
#    def init():
#        for i in range(len(shape) - 1):
#            #ww[i] = np.random.random((shape[i] + 1, shape[i + 1]))
#            ww[i] = np.zeros((shape[i] + 1, shape[i + 1]))
#
#        for i in range(len(shape)):
#            ss[i] = np.zeros(shape[i])
#            aa[i] = np.zeros(shape[i] + 1)
#
#
#    def logistic(x):
#        return 1.0 / (1 + np.exp(-x))
#
#    def logisticd(x):
#        return logistic(x) * (1 - logistic(x))
#
#    vlogistic = np.vectorize(logistic)
#    vlogisticd = np.vectorize(logisticd)
#
#    def forward(x):
#        aa[0][0] = -1.0
#        aa[0][1 : 1 + dim] = x
#
#        for i in range(1, len(shape)):
#            ss[i] = aa[i - 1].dot(ww[i - 1])
#
#            aa[i][0] = -1.0
#            aa[i][1 : 1 + shape[i]] = vlogistic(ss[i])
#
#        return aa[-1][1:]
#
#    def J():
#        s = 0.0
#
#        for x, y in zip(datax, datay):
#            hx = forward(x)
#            for i in range(2):
#                if y[i] == 0:
#                    s -= (1 - y[i]) * np.log(1 - hx[i])
#                else:
#                    s -= y[i] * np.log(hx[i])
#
#        s /= len(datax)
#
#        reg = 0.0
#        for w in ww:
#            reg += np.linalg.norm(w) ** 2
#        reg = reg * lam / (2 * len(datax))
#
#        return s + reg
#
#    def backprop(correct):
#        deltas = aa[-1][1:] - correct
#        for i in range(len(shape) - 1, 0, -1):
#            #print("------")
#            #print("Deltas: {}".format(deltas))
#            #print("vlog: {}".format(vlogisticd(ss[i])))
#            #print("------")
#            deltas = np.multiply(deltas, vlogisticd(ss[i]))
#            pdeltas = deltas.dot(np.transpose(ww[i - 1][1:][:]))
#            dw = np.outer(aa[i - 1], deltas)
#            ww[i - 1] -= 0.1 * dw
#            deltas = pdeltas
#
#
#    init()
#
#    for i in range(200): # TODO
#        #if i % 5 == 0:
#        #    print("Step {}: value {}".format(i, J()))
#        for x, y in zip(datax, datay):
#            res = forward(x)
#            backprop(y)
#    print("--------------")
#    for w in ww:
#        print(w)
#    print("--------------")
#
#    def classify(x):
#        y = forward(x)
#        d1 = np.linalg.norm(y - NEGATIVE)
#        d2 = np.linalg.norm(y - POSITIVE)
#        return -1 if d1 < d2 else 1
#
#    return classify

def test_neural_network(test_set, classifier):
    res = []
    for e in test_set:
        #print("Point {}".format(e.features))
        r = classifier(e.features)
        # print("Result {}".format(r))
        res.append(r)
    return res