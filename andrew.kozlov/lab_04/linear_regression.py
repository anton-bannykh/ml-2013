from math import exp
from numpy.linalg import norm
from numpy.ma import inner, zeros, array, average

__author__ = 'adkozlov'


def classify(x, w):
    v = -inner(x, w)
    return 1.0 if 1 / (1 + exp(v)) >= 0.5 else -1.0


def calculate_error(data, w):
    count = 0

    for (x, y) in data:
        if classify(x, w) != y:
            count += 1

    return count / len(data)


def line_length(data):
    x, _ = data[0]
    return len(x)


def linear_regression_w(data, c, exp_value=20, eps=0.1, rate=0.01):
    m = line_length(data)
    w = zeros(m)

    dif_prev = 0
    while True:
        w_prev = array(w)

        for (x, y) in data:
            grad = zeros(m)

            for j in range(m):
                v = y * inner(w, x)

                if v < exp_value:
                    grad[j] += y * x[j] / (1 + exp(v))
                if j != 0:
                    grad[j] += c * w[j]

            w += rate * grad

        dif_norm = norm(w - w_prev)
        if dif_prev < dif_norm and dif_prev != 0:
            break
        else:
            dif_prev = dif_norm
        if dif_norm < eps:
            break

    return w


def cross_validate(data, c, n=10):
    step = len(data) // n
    errors = []

    for i in range(0, len(data), step):
        w = linear_regression_w(data[i + step:] + data[:i], c)
        errors.append(calculate_error(data[i:i + step], w))

    return average(errors)


def optimize_constant(data, n=10):
    result, error = 0, 1

    for d in range(n):
        c = 0.5 ** d
        average_error = cross_validate(data, c)

        if error > average_error or result == 0:
            error = average_error
            result = c

    return result
