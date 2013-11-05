from collections import namedtuple
import itertools
import os
import os.path
import random
import sys
from sys import stdin, stdout, stderr

import numpy
import numpy.linalg

DATA_DIR = "data"
TMP_DIR = "tmp"

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

DATA_LOCAL_NAME = "wdbc.data"
DATA_LOCAL_PATH = os.path.join(TMP_DIR, DATA_LOCAL_NAME)

TRAIN_SET_NAME = "train_set"
TRAIN_SET_PATH = os.path.join(DATA_DIR, TRAIN_SET_NAME)

TEST_SET_NAME = "test_set"
TEST_SET_PATH = os.path.join(DATA_DIR, TEST_SET_NAME)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

# Dataset entry
Entry = namedtuple("Entry", ["id", "correct", "features"])

class Result:
    FN = "False negative"
    FP = "False positive"
    TN = "True negative"
    TP = "True positive"

# downloads data set to a temporary file
def get_data(url, local_path):
    import urllib.request
    import shutil

    with urllib.request.urlopen(url) as fu, open(local_path, 'wb') as f:
        shutil.copyfileobj(fu, f)

# loads data set from the temporary file and converts it appropriately
def load_data(local_path):
    res = []
    with open(local_path) as f:
        for line in f.readlines():
            l = line.split(',')
            id = int(l[0])
            diagnosis = -1 if l[1] == 'B' else 1
            ff = [float(x) for x in l[2: 32]]
            ff.append(0.0) # bias
            features = numpy.array(ff)
            res.append(Entry(id = id, correct = diagnosis, features = features))
    return res

# exact inverse of the 'load_data' function
def write_data(data_set, local_path):
    with open(local_path, 'w') as f:
        for de in data_set:
            l = []
            l.append(str(de.id))
            l.append('B' if de.correct == -1 else 'M')
            l.extend([str(x) for x in de.features])
            f.write(','.join(l) + "\n")

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


def train_perceptron(train_set, initial):
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

#get_data(DATA_URL, DATA_LOCAL_PATH)
#stderr.write("Data set fetched to the file {}\n".format(DATA_LOCAL_PATH))

data = load_data(DATA_LOCAL_PATH)
data_size = 569
assert len(data) == data_size # there should be 569 instances according to the set description

test_size = int(569 * 0.20)
train_size = data_size - test_size
train_set, test_set = split_data(data, train_size, test_size)

stderr.write("Train set size: {}\n".format(train_size))
stderr.write("Test set size: {}\n".format(test_size))

write_data(train_set, TRAIN_SET_PATH)
stderr.write("Train set dumped to the file {}\n".format(TRAIN_SET_PATH))
write_data(test_set, TEST_SET_PATH)
stderr.write("Test set dumped to the file {}\n".format(TEST_SET_PATH))

initial = numpy.zeros(31)
theta = initial

for i in range(100):
    theta = train_perceptron(train_set, theta)
    test_ans = test_perceptron(test_set, theta)
    results = calculate_results(test_set, test_ans)
    print("Step {}: classification error is {}%".format(i, calculate_error_rate(results) * 100))

print(theta)
print("Classification error is {}%".format(calculate_error_rate(results) * 100))
print("Classification precision is {}%".format(calculate_precision(results) * 100))
print("Classification recall is {}%".format(calculate_recall(results) * 100))