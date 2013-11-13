import numpy as np
from cvxopt import matrix
from cvxopt.blas import dotu
from cvxopt.solvers import qp

__author__ = 'adkozlov'


def classify(x, w, b):
    return 1 if np.dot(w, x) + b > 0 else -1


def vector(n, i, v=1):
    result = [0 for _ in range(2 * n)]

    result[2 * i] = v
    result[2 * i + 1] = -v

    return result


def kernel(xs, ys):
    return dotu(matrix(xs), matrix(ys))


def matrices(data, c):
    p = [[y1 * y2 * kernel(x1, x2) for (x2, y2) in data] for (x1, y1) in data]

    q = [-1.0 for _ in data]

    n = len(data)
    g = []
    for i in range(n):
        g.append(vector(n, i))

    h = []
    for i in range(n):
        h.append(c)
        h.append(0.0)
    h = [h]

    a = [[y] for (_, y) in data]

    b = [0.0]

    return [matrix(m) for m in [p, q, g, h, a, b]]


def fit_svm(data, c, eps=1e-6):
    solution = qp(*matrices(data, c))

    x0, _ = data[0]
    m = len(x0)

    a = np.array(solution['x']).transpose()[0]
    w = np.zeros(m)

    for i in range(len(data)):
        if a[i] < eps:
            continue

        x, y = data[i]
        w += y[i] * a[i] * np.array(x[i])

    b = 0
    for i in range(len(data)):
        if 2 * eps < a[i] + eps < c:
            x, y = data[i]
            b = y[i] - np.dot(w, x[i])
            break

    return w, b


def calculate_error(data, w, b):
    count = 0

    for (x, y) in data:
        if classify(x, w, b) != y:
            count += 1

    return count / len(data)


def cross_validate(data, c, n=10):
    step = len(data) // n
    errors = []

    for i in range(0, len(data), step):
        w, b = fit_svm(data[i + step:] + data[:i], c)
        errors.append(calculate_error(data[i:i + step], w, b))

    return np.average(errors)


def optimize_c(data, n=10):
    result, error = 0, 0

    for d in range(-n // 2, n // 2):
        c = 10 ** d
        average_error = cross_validate(data, c)

        if error > average_error or result == 0:
            error = average_error
            result = c

    return result