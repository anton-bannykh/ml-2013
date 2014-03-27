__author__ = 'charlie'

import numpy
import utils


def function(x, y):
    if numpy.dot(x, y) >= 0:
        return 1
    else:
        return -1


def train(x, y, count):
    size = len(x)
    block_size = len(x[0])
    vector = numpy.zeros(block_size)
    for i in range(count):
        for j in range(size):
            if function(vector, x[j]) != y[j]:
                vector += y[j] * x[j]
    return vector


def test(x, y, v):
    res = [numpy.zeros(2), numpy.zeros(2)]
    for i in range(len(x)):
        vector = function(v, x[i])
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


def run(train_x, train_y, test_x, test_y, steps=1000):
    vector = train(train_x, train_y, steps)
    # print(vector)
    res = test(test_x, test_y, vector)
    return utils.process_result(res)