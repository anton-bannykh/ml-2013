from math import exp
import numpy

__author__ = 'adkozlov'


def norm(w):
    return numpy.sqrt(numpy.inner(w, w))


def function(x, c=1):
    return c / (1 + exp(x))


def classify(x, w, margin=0.5):
    return 1.0 if function(-numpy.inner(x, w)) >= margin else -1.0


def calculate_error(data, w):
    count = 0

    for (x, y) in data:
        if classify(x, w) != y:
            count += 1

    return count / len(data)


def line_len(data):
    x, _ = data[0]
    return len(x)


def linear_regression_w(data, c, exp_value=20, eps=0.1, rate=0.01):
    m = line_len(data)
    w = numpy.zeros(m)

    dif_prev = 0
    while True:
        w_prev = numpy.array(w)

        for (x, y) in data:
            grad = numpy.zeros(m)

            for j in range(m):
                v = y * numpy.inner(w, x)

                if v < exp_value:
                    grad[j] += function(v, y * x[j])
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
    result = 0

    for i in range(0, len(data), step):
        w = linear_regression_w(data[i + step:] + data[:i], c)
        result += calculate_error(data[i:i + step], w)

    return result / n


def optimize_constant(data, base=3, n=10):
    result, error = 1, 1

    for d in range(n):
        c = base ** -d
        average_error = cross_validate(data, c)
        print("current constant = %f, current error = %f" % (c, average_error))

        if average_error < error:
            error = average_error
            result = c

    return result
