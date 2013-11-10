import os
from sys import stdin, stdout, stderr

import numpy

from bcwd import *
from perceptron import *

DATA_DIR = "data"
TMP_DIR = "tmp"

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

DATA_LOCAL_NAME = "wdbc.data"
DATA_LOCAL_PATH = os.path.join(TMP_DIR, DATA_LOCAL_NAME)

DATA_SIZE = 569 # according to the set description

TRAIN_SET_NAME = "train_set"
TRAIN_SET_PATH = os.path.join(DATA_DIR, TRAIN_SET_NAME)

TEST_SET_NAME = "test_set"
TEST_SET_PATH = os.path.join(DATA_DIR, TEST_SET_NAME)

TEST_SET_FRACTION = 0.10

TRAINING_ITERATIONS = 100

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

get_data(DATA_URL, DATA_LOCAL_PATH)
stderr.write("Data set fetched to the file {}\n".format(DATA_LOCAL_PATH))
stderr.write("Comment the line 35 of main.py to prevent downloading during the further runs\n")

data = load_data(DATA_LOCAL_PATH)
assert len(data) == DATA_SIZE # there should be 569 instances according to the set description


def run_all(it, data, initial, iterations, verbose = False):
    test_size = int(DATA_SIZE * TEST_SET_FRACTION)
    train_size = DATA_SIZE - test_size
    train_set, test_set = split_data(data, train_size, test_size)

    if verbose:
        stderr.write("Train set size: {}\n".format(train_size))
    if verbose:
        stderr.write("Test set size: {}\n".format(test_size))

    write_data(train_set, TRAIN_SET_PATH)
    if verbose:
        stderr.write("Train set dumped to the file {}\n".format(TRAIN_SET_PATH))
    write_data(test_set, TEST_SET_PATH)
    if verbose:
        stderr.write("Test set dumped to the file {}\n".format(TEST_SET_PATH))

    theta = initial

    for i in range(iterations):
        theta = train_perceptron_step(train_set, theta)
        if verbose:
            test_ans = test_perceptron(test_set, theta)
            results = calculate_results(test_set, test_ans)
            stderr.write("Step {}: classification error is {}%\n".format(i, calculate_error_rate(results) * 100))

    test_ans = test_perceptron(test_set, theta)
    results = calculate_results(test_set, test_ans)

    err_rate = calculate_error_rate(results)
    precision = calculate_precision(results)
    recall = calculate_recall(results)

    return (theta, err_rate, precision, recall)

# random.seed(0) # uncomment to make the program deterministic

cnt = 100
s = 0.0

for i in range(cnt):
    stderr.write("Running... {}/{}\n".format(i, cnt))
    initial = numpy.zeros(31)
    theta, err_rate, prec, rec = run_all(cnt, data, initial, TRAINING_ITERATIONS, verbose = False)
    s += err_rate

print("Average error rate ({} runs) is {}%".format(cnt, s / cnt * 100))