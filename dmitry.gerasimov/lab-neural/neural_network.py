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

def test_neural_network(test_set, classifier):
    res = []
    for e in test_set:
        #print("Point {}".format(e.features))
        r = classifier(e.features)
        # print("Result {}".format(r))
        res.append(r)
    return res