from collections import namedtuple
import random

import numpy

# Dataset entry
Entry = namedtuple("Entry", ["id", "correct", "features"])

class Result:
    FN = "False negative"
    FP = "False positive"
    TN = "True negative"
    TP = "True positive"


# splits data set in training set, test set and validation srt according of sizes test_cnt, train_cnt and val_cnt
def split_data(data_set, train_cnt, test_cnt, val_cnt):
    if train_cnt + test_cnt + val_cnt > len(data_set):
        raise AttributeError("train_cnt + test_cnt + val_cnt should be no more than data set size")
    ds = [x for x in data_set]
    random.shuffle(ds)
    train_set = ds[0: train_cnt]
    test_set = ds[train_cnt: train_cnt + test_cnt]
    val_set = ds[train_cnt + test_cnt: train_cnt + test_cnt + val_cnt]
    return (train_set, test_set, val_set)


# appends 1.0 to all feature vectors
def add_bias(data_set):
    ud = []
    for d in data_set:
        ud.append(Entry(id = d.id, correct = d.correct, features = numpy.append(d.features, 1.0)))
    return ud

def error_rate(results):
    fp = results.count(Result.FP)
    fn = results.count(Result.FN)
    return (fp + fn) / len(results)


def precision(results):
    tp = results.count(Result.TP)
    fp = results.count(Result.FP)
    if tp == 0 and fp == 0:
        # all predictions are negative
        return 1.0
    else:
        return tp / (tp + fp)


def recall(results):
    tp = results.count(Result.TP)
    fn = results.count(Result.FN)
    if tp == 0 and fn == 0:
        # no positives in data
        return 1.0
    return tp / (tp + fn)


def f1score(results):
    pr = precision(results)
    re = recall(results)
    if pr == 0 and re == 0:
        return 0.0
    else:
        return 2 * pr * re / (pr + re)

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
