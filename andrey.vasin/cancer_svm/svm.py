from cvxopt import matrix, solvers
from cancer_svm.kernel import *
import numpy

def transpose(ls):
    return list(map(list, zip(*ls)))

def get_lagrange_coef_reg(x, y, const, kernel):
    solvers.options['show_progress'] = False
    P = numpy.identity(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            #P[i][j] = y[i] * y[j] * numpy.inner(x[i], x[j])
            P[i][j] = y[i] * y[j] * kernel(x[i], x[j])
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
    lagrange = [sol[i] for i in range(len(x))]
    return lagrange

def get_w(x, y, C):
    lagrange = get_lagrange_coef_reg(x, y, C, inner_product_kernel)

    eps = 1e-5

    w = numpy.zeros(len(x[0]))
    for i in range(len(x)):
        if lagrange[i] > eps:
            w += lagrange[i] * y[i] * numpy.array(x[i])

    b = 0
    for i in range(len(x)):
        if eps < lagrange[i] < C - eps:
            b = y[i] - numpy.inner(w, x[i])
            break

    return w, b

def kernel_predict(x_out, x_in, y_in, lagrange, kernel):
    eps = 1e-5

    b = 0
    for i in range(len(x_in)):
        if lagrange[i] > eps:
            b += y_in[i]
            for j in range(len(x_in)):
                if lagrange[j] > eps:
                    b -= lagrange[j] * y_in[j] * kernel(x_in[i], x_in[j])
            break

    y_out = []
    for x_out_cur in x_out:
        y_cur = 0
        for i in range(len(x_in)):
            if lagrange[i] > eps:
                y_cur += lagrange[i] * y_in[i] * kernel(x_in[i], x_out_cur)
        y_cur += b
        y_out.append(1.0 if y_cur > 0 else -1.0)

    return y_out


def predict(x, w, b):
    return [1.0 if numpy.inner(w, x[i]) + b > 0 else -1.0 for i in range(len(x))]