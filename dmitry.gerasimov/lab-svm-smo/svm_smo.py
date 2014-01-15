from cvxopt import matrix, solvers
import numpy as np

from common import *

# phi : vector ->  float
# K(x, y) = phi(x) * phi(y)
# returns a classifier function
def train_svm(training_set, C, K):
    datax = [x.features for x in training_set]
    datay = [x.correct for x in training_set]


    solvers.options['show_progress'] = False

    m = len(training_set) # number of training examples
    dim = len(datax[0]) # dimension of the feature vector

    # explanation of this weird stuff is at <http://cvxopt.org/userguide/coneprog.html#quadratic-programming>
    # we optimize min 1/2 * \sum_i \sum_j y_i y_j K(x_i, x_j) a_i a_j - \sum_i a_i
    pP = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            pP[i][j] = datay[i] * datay[j] * K(datax[i], datax[j])
    P = matrix(pP)

    pq = np.zeros([m, 1])
    for i in range(m):
        pq[i][0] = -1.0
    q = matrix(pq)

    # we have m constraints of form: 0 <= a_i <= C, therefore:
    # m constraints of form -a_i <= 0
    # m constraints of form a_i <= C

    pG = np.zeros([2 * m, m])
    ph = np.zeros([2 * m, 1])
    for i in range(m):
        pG[i][i] = -1.0
        ph[i] = 0.0 # just for clarity

    for i in range(m):
        pG[m + i][i] = 1.0
        ph[m + i] = C

    G = matrix(pG)
    h = matrix(ph)

    # One equality constraint: \sum_i a_i y_i = 0
    pA = np.zeros([1, m])
    pb = np.zeros([1, 1])
    for i in range(m):
        pA[0][i] = datay[i]

    pb[0][0] = 0.0 # just for clarity

    A = matrix(pA)
    b = matrix(pb)

    sol = solvers.qp(P, q, G, h, A, b)['x']

    # now we can find the bias using a support vector x_i such that 0 < a_i < C
    # the only problem is we might have numbers indistinguishable from zero
    tolerance = 1e-6

    b = None
    for i in range(m):
        if 0 + tolerance < sol[i] < C - tolerance:
            s = 0.0
            for j in range(m):
                s += sol[j] * datay[j] * K(datax[j], datax[i])
            b = datay[i] - s
            break

    def classify(v):
        s = 0.0
        for i in range(m):
            s += sol[i] * datay[i] * K(v, datax[i])
        s += b
        if s >= 0:
            return 1
        else:
            return -1

    return classify


def test_svm(test_set, classifier):
    res = []
    for e in test_set:
        #print("Point {}".format(e.features))
        r = classifier(e.features)
        #print("Result {}".format(r))
        res.append(r)
    return res