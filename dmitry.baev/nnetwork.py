__author__ = 'charlie'

import numpy
import math
import utils


def activation_func(x):
    return 1.0 / (1.0 + math.exp(-x))


def activate(data):
    return [activation_func(x) for x in data]


def function(edge_weight, active):
    l = len(edge_weight[0])
    result = numpy.zeros(l)
    for j in range(l):
        for i in range(len(active)):
            result[j] += edge_weight[i][j] * active[i]
    return result


def find_best_c(x, y, share, count):
    x_train, x_check = utils.split_data(x, share)
    y_train, y_check = utils.split_data(y, share)

    best_f1 = 0
    best_c = -1
    c = 10
    while c <= 40:
        w1, w2 = train(x_train, y_train, c, count)
        p, r = utils.process_result(test(x_check, y_check, w1, w2))
        f1 = utils.f1(p, r)
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
        c += 10
    return best_c


def train(x_origin, y, c, count):
    g = 0.8
    l = len(x_origin[0])
    x = [numpy.append(i, [1.]) for i in x_origin]
    ew1 = utils.random_array(l + 1, c)
    ew2 = utils.random_array(c + 1, 1)
    for times in range(count):
        dew1 = [numpy.zeros(c) for tmp in range(l + 1)]
        dew2 = [0 for tmp in range(c + 1)]
        for t in range(len(x)):
            active1 = activate(function(ew1, x[t]))
            active1 = numpy.append(active1, 1.0)
            active2 = activate(function(ew2, active1))
            d2 = -active2[0] * (1. - active2[0]) * (active2[0] - (y[t] + 1.) / 2.)
            d1 = [active1[i] * (1. - active1[i]) * ew2[i][0] * d2 for i in range(c)]
            for i in range(c + 1):
                dew2[i] += g * active1[i] * d2
            for k in range(l + 1):
                for j in range(c):
                    dew1[k][j] += g * x[t][k] * d1[j]
        for i in range(c + 1):
            ew2[i][0] += dew2[i]
        for i in range(l + 1):
            for j in range(c):
                ew1[i][j] += dew1[i][j]
    return ew1, ew2


def test(x_origin, y, edge_weights1, edge_weights2):
    x = [numpy.append(i, [1]) for i in x_origin]
    res = [numpy.zeros(2), numpy.zeros(2)]
    for i in range(len(x)):
        active1 = activate(function(edge_weights1, x[i]))
        active1 = numpy.append(active1, [1])
        active2 = activate(function(edge_weights2, active1))

        if y[i] == 1.0:
            if active2[0] > 0.5:
                res[0][0] += 1
            else:
                res[0][1] += 1
        else:
            if active2[0] > 0.5:
                res[1][0] += 1
            else:
                res[1][1] += 1
    return res


def run(train_x, train_y, test_x, test_y, share=0.4, count=100):
    best_c = find_best_c(train_x, train_y, share, count)
    print(best_c)
    w1, w2 = train(train_x, train_y, best_c, count)
    print(w2)
    res = test(test_x, test_y, w1, w2)
    print(res)
    return utils.process_result(res)

