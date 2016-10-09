import numpy as np
from math import exp, log
from scipy import optimize

from common import *

# returns a classifier function
# C is the regularisation constand, lambda is not gonna work in python
def train_logistic(training_set, C):
    datax = [x.features for x in training_set]
    datay = [x.correct for x in training_set]
    datay = [1 if y == 1 else 0 for y in datay] # negative prob is 0.0

    dim = len(datax[0])

    def sigmoid(z):
        #print(z)
        r = 1.0 / (1.0 + np.exp(-z))
        #print(r)
        return r

    def h(theta, x):
        return sigmoid(np.dot(theta, x))

    def safe_log(x):
        if x <= 0:
            return 0.0 # the function is non-continuous now :(( However, seems to be working
        else:
            return np.log(x)

    def J(theta):
        l = [-y * safe_log(h(theta, x)) - (1 - y) * safe_log(1 - h(theta, x)) for x, y in zip(datax, datay)]
        s = sum(l) / len(datax) + C * np.linalg.norm(theta) / (2 * len(datax))
        return s

    def dJ(theta):
        d = [None for _ in range(dim)]
        hx = [h(theta, x) for x in datax]
        for j in range(dim):
            d[j] = sum([(hx[i] - datay[i]) * datax[i][j] for i in range(len(datax))]) + C * theta[j]
            d[j] /= len(datax)
        return np.array(d)


    theta = optimize.fmin_cg(J, np.zeros(dim), fprime = dJ, maxiter = 300, disp = 0)
    #print(theta)

    def classify(v):
        prob = h(theta, v)
        if prob >= 0.5:
            return 1
        else:
            return -1

    return classify


def test_logistic(test_set, classifier):
    res = []
    for e in test_set:
        #print("Point {}".format(e.features))
        r = classifier(e.features)
        #print("Result {}".format(r))
        res.append(r)
    return res