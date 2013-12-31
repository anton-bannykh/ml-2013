import random

import numpy

from common import *

# splits data set in test set and training set according of sizes test_cnt and train_cnt accordingly
def split_data(data_set, train_cnt, test_cnt):
    if train_cnt + test_cnt > len(data_set):
        raise AttributeError("train_cnt + test_cnt should be no more than data set size")
    ds = [x for x in data_set]
    random.shuffle(ds)
    train_set = ds[0: train_cnt]
    test_set = ds[train_cnt: train_cnt + test_cnt]
    return (train_set, test_set)

def f(x, theta):
    dp = numpy.dot(x, theta)
    return 1 if dp > 0 else -1


def train_perceptron_step(train_set, initial):
    theta = numpy.copy(initial)
    for e in train_set:
        val = f(e.features, theta)
        if val != e.correct:
            theta += e.features * e.correct

    return theta

def test_perceptron(test_set, theta):
    res = []
    for e in test_set:
        res.append(f(e.features, theta))
    return res

def calculate_results(test_set, test_ans):
    results = []
    for cur, e in zip(test_ans, test_set):
        if e.correct == -1:
            if cur == -1:
                results.append(Result.TN)
            else:
                results.append(Result.FP)
        else:
            if cur == -1:
                results.append(Result.FN)
            else:
                results.append(Result.TP)
    return results