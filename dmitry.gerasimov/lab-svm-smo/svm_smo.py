from cvxopt import matrix, solvers
import numpy as np

from common import *

def f(features, b, w):
    return np.dot(w, features) + b

# phi : vector ->  float
# K(x, y) = phi(x) * phi(y)
def train_svm(training_set, C, phi):
    def K(x, y):
        return np.dot(phi(x), phi(y))

    solvers.options['show_progress'] = False

    m = len(training_set) # number of training examples
    dim = len(training_set[0].features) # dimension of the feature vector

    # explanation of this weird stuff is at <http://cvxopt.org/userguide/coneprog.html#quadratic-programming>
    # we optimize min 1/2 * \sum_i \sum_j y_i y_j K(x_i, x_j) a_i a_j - \sum_i a_i
    pP = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            pi = training_set[i]
            pj = training_set[j]
            pP[i][j] = pi.correct * pj.correct * K(pi.features, pj.features)
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
        pA[0][i] = training_set[i].correct

    pb[0][0] = 0.0 # just for clarity

    A = matrix(pA)
    b = matrix(pb)

    sol = solvers.qp(P, q, G, h, A, b)['x']

    # we have a's, so we should extract w and b now

    w = np.zeros(dim)
    for i in range(m):
        w += sol[i] * training_set[i].correct * training_set[i].features

    # now we have to find such an i that 0 < a_i < C, in that case: y_i (w^T phi(x_i) + b) = 1
    # the problem is we might have numbers indistinguishable from zero
    # we are going to find
    tolerance = 1e-6

    b = None
    for i in range(m):
        if 0 + tolerance < sol[i] < C - tolerance:
            b = training_set[i].correct - np.dot(w, phi(training_set[i].features))
            break

    return b, w


def test_svm(test_set, b, theta):
    res = []
    for e in test_set:
        #print("Point {}".format(e.features))
        r = f(e.features, b, theta)
        #print("Result {}".format(r))
        rr = None
        if r >= 1:
            rr = 1
        else:
            rr = -1
        res.append(rr)
    return res