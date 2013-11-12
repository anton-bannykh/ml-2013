import numpy

from common import *

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

def calculate_error_rate(results):
    fp = results.count(Result.FP)
    fn = results.count(Result.FN)
    return (fp + fn) / len(results)

def calculate_precision(results):
    tp = results.count(Result.TP)
    fp = results.count(Result.FP)
    return tp / (tp + fp)

def calculate_recall(results):
    tp = results.count(Result.TP)
    fn = results.count(Result.FN)
    return tp / (tp + fn)