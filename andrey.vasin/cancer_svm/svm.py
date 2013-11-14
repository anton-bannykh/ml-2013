from cvxopt import matrix, solvers
import cvxopt.blas
import numpy

def transpose(ls):
    return list(map(list, zip(*ls)))

def get_lagrange_coef_reg(x, y, const):
    solvers.options['show_progress'] = False
    P = numpy.identity(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            P[i][j] = y[i] * y[j] * numpy.inner(x[i], x[j])
    P = matrix(P)
    A = matrix(transpose([y]))
    b = matrix([0.0])
    q = matrix([-1.0] * len(x))
    G = numpy.zeros(shape=(len(x) * 2, len(x)))
    h = matrix([0.0] * len(x) + [const] * len(x))
    for i in range(len(x)):
            G[i][i] = -1
    for i in range(len(x)):
            G[len(x) + i][i] = 1
    G = matrix(transpose(G))
    sol = solvers.qp(P, q, G, h, A, b)['x']
    alpha = [sol[i] for i in range(len(x))]
    return alpha

def get_w(x, y, C):
    alpha = get_lagrange_coef_reg(x, y, C)

    eps = 1e-5

    w = numpy.zeros(len(x[0]))
    for i in range(len(x)):
        if alpha[i] > eps:
            w += alpha[i] * y[i] * numpy.array(x[i])

    b = 0
    for i in range(len(x)):
        if eps < alpha[i] < C - eps:
            b = y[i] - numpy.inner(w, x[i])
            break

    return w, b

def predict(x, w, b):
    return [1.0 if numpy.inner(w, x[i]) + b > 0 else -1.0 for i in range(len(x))]