__author__ = 'charlie'

import numpy
import perceptrone
import svm
import nnetwork
import utils


def parse_data(data):
    first = [numpy.array([float(i) for i in x[2:]]) for x in data]
    n_first = normalize(first)
    second = [1.0 if x[1] == 'M' else -1.0 for x in data]
    return n_first, second


def init_data(share):
    data_lines = open('wdbc.data').readlines()
    data = [x.split(',') for x in data_lines]
    # numpy.random.shuffle(data)
    return utils.split_data(data, share)


def normalize(data):
    for i in range(len(data[0])):
        tmp = [x[i] for x in data]
        minimum = min(tmp)
        maximum = max(tmp)
        m = float(minimum + maximum) / 2.0
        r = float(maximum - minimum) / 2.0
        for j in data:
            j[i] = (j[i] - m) / r
    return data


def print_result(name, precision, recall):
    print(name)
    print("precision: %.3f recall: %.3f" % (precision, recall))
    print("f1: %.3f" % (utils.f1(precision, recall)))
    print("###############")


#
train, test = init_data(0.4)
train_x, train_y = parse_data(train)
test_x, test_y = parse_data(test)

# print(train_x[0])

# prec_precision, prec_recall = perceptrone.run(train_x, train_y, test_x, test_y)
# print_result("perceptrone", prec_precision, prec_recall)

# svm_precision, svm_recall = svm.run(train_x, train_y, test_x, test_y)
# print_result("svm", svm_precision, svm_recall)

nn_precision, nn_recall = nnetwork.run(train_x, train_y, test_x, test_y)
print_result("neural network", nn_precision, nn_recall)
#
# a = [[1, 2, 4], [3, 4, 7]]
# b = normalize(a)
# print(b)

# data = [x for x in data_lines]


