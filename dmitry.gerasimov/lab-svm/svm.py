from cvxopt import matrix, solvers
import numpy

from common import *

def f(features, b, w):
    return sum([a * b for a, b in zip(features, w)]) + b

def train_svm(training_set, C):
    solvers.options['show_progress'] = False

    m = len(training_set) # number of training examples
    dim = 30 # dimension of the feature vector

    # explanation of this weird stuff is at <http://cvxopt.org/userguide/coneprog.html#quadratic-programming>
    # structure of variables:
    # 0: bias
    # [1, 1 + dim): weights
    # [1 + dim, 1 + dim + m): regularisation variables (xis)

    pP = [[0.0 for _ in range(1 + dim + m)] for _ in range(1 + dim + m)]
    # set ||w||^2 constraint
    for i in range(1, 1 + dim):
        pP[i][i] = 1.0
    P = matrix(pP)

    pq = [0.0 for _ in range(1 + dim + m)]
    for i in range(1 + dim, 1 + dim + m):
        pq[i] = C
    q = matrix(pq)


    pG = [[0.0 for _ in range(1 + dim + m)] for _ in range(2 * m)]
    ph = [0.0 for i in range(2 * m)]

    # set \xi_i >= 0 constraints
    for i in range(0, m):
        pG[i][1 + dim + i] = -1.0
        # no need to set right hand side

    # set constraints on training set points
    # y_i (w^T x_i + b) >= 1 - \xi_i
    for i in range(0, m):
        ph[m + i] = -1.0 # rhs

        point = training_set[i]
        line = pG[i + m]
        # bias coefficient
        line[0] = -point.correct
        # dot product coeffecients
        for j, f in enumerate(point.features):
            line[1 + j] = -point.correct * f
        # regularisation coeffecient
        line[1 + dim + i] = -1.0

    G = matrix(([[pG[i][j] for i in range(2 * m)] for j in range(1 + dim + m)])) # cvxopt uses column-major order
    h = matrix(ph)

    # no equality constraints

    sol = solvers.qp(P, q, G, h)['x']
    b = sol[0]
    w = [sol[i] for i in range(1, 1 + dim)]
    xi = [sol[i] for i in range(1 + dim, 1 + dim + m)]
    return b, w, xi


def test_svm(test_set, b, theta):
    res = []
    for e in test_set:
        r = f(e.features, b, theta)
        rr = None
        if r >= 1:
            rr = 1
        else:
            rr = -1
        res.append(rr)
    return res