__author__ = 'charlie'

import numpy
import utils
from scipy.optimize import minimize


def function(x, y, x0):
    if numpy.dot(x, y) + x0 >= 0:
        return 1
    else:
        return -1


def find_best_c(x, y, share):
    x_train, x_check = utils.split_data(x, share)
    y_train, y_check = utils.split_data(y, share)

    best_c = 2 ** -7
    best_f1 = 0
    for i in range(-7, 7):
        c = 2 ** i
        v = train(x_train, y_train, c)
        p, r = utils.process_result(test(x_check, y_check, v))
        f1 = utils.f1(p, r)
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
    return best_c


def train(x, y, c):
    def f(tmp):
        t, t0 = tmp[:-1], tmp[-1]
        s = 0
        for i in range(len(y)):
            s += numpy.maximum(0, 1 - y[i] * (numpy.dot(t, x[i]) + t0))
        return 0.5 * sum(t ** 2) + c * s

    return minimize(f, numpy.ones(len(x[0]) + 1)).x


def test(x, y, v):
    vec, c = v[:-1], v[-1:]
    res = [numpy.zeros(2), numpy.zeros(2)]
    for i in range(len(x)):
        vector = function(vec, x[i], c)
        # print(vector)
        if y[i] == 1.0:
            if vector > 0:
                res[0][0] += 1
            else:
                res[0][1] += 1
        else:
            if vector > 0:
                res[1][0] += 1
            else:
                res[1][1] += 1
    return res


def run(train_x, train_y, test_x, test_y, test_share=0.4):
    best_c = find_best_c(train_x, train_y, test_share)
    vector = train(train_x, train_y, best_c)
    # print(vector)
    res = test(test_x, test_y, vector)
    return utils.process_result(res)