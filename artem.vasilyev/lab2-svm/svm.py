import cvxopt
from cvxopt import solvers
import numpy as np


def learn(X, Y, C):
    assert len(X) == len(Y)

    dimX = len(X[0])
    dim = dimX + 1 + len(X)

    P = cvxopt.spmatrix(1.0, range(dimX), range(dimX), (dim, dim))
    q = cvxopt.matrix([0] * (dimX + 1) + [C] * len(X), tc='d')
    constraints = len(X) * 2

    gMatrix = np.ndarray((constraints, dim))
    for i in range(len(X)):
        gMatrix[i] = np.concatenate((-Y[i] * X[i], [-Y[i]], np.repeat(-1, len(X))))

    for i in range(len(X)):
        gMatrix[i + len(X)] = np.repeat(0, dim)
        gMatrix[i + len(X)][dimX + 1 + i] = 1

    G = cvxopt.matrix(gMatrix, tc='d')
    h = cvxopt.matrix(np.concatenate((np.repeat(-1, len(X)), np.repeat(0, len(X)))), tc='d')

    result = solvers.qp(P, q, G, h)['x']
    theta = result[:dimX].T
    theta0 = result[dimX]

    print(theta, theta0)

    return lambda t: classify(t, theta, theta0)

def classify(x, theta, theta0=0):
    val = np.inner(x, theta) + theta0
    return 1 if val >= 0 else -1


