__author__ = 'charlie'

import random


def f1(precision, recall):
    return 2 * precision * recall / (recall + precision)


def split_data(data, share):
    size = int(len(data) * share)
    return [x for x in data[:size]], [x for x in data[size:]]


def process_result(res):
    precision = res[0][0] / (res[0][0] + res[0][1])
    recall = res[0][0] / (res[0][0] + res[1][0])
    return precision, recall


def random_array(*args):
    if len(args) == 0:
        return random.random()
    else:
        return [random_array(*args[1:]) for i in range(args[0])]