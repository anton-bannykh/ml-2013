import os
from sys import stdin, stdout, stderr

import numpy

from bcwd import *
from svm import *

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

VALID_SET_NAME = "valid.set"
VALID_SET_PATH = os.path.join(DATA_DIR, VALID_SET_NAME)

TEST_SET_FRACTION = 0.10
VALID_SET_FRACTION = 0.10

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

get_data(DATA_URL, DATA_LOCAL_PATH)
stderr.write("Data set fetched to the file {}\n".format(DATA_LOCAL_PATH))
stderr.write("Comment the line 39 of main.py to prevent downloading during the further runs\n")

data = load_data(DATA_LOCAL_PATH)
assert len(data) == DATA_SIZE # there should be 569 instances according to the set description


def run_all(data, verbose = False):
    test_size = int(DATA_SIZE * TEST_SET_FRACTION)
    valid_size = int(DATA_SIZE * VALID_SET_FRACTION)
    train_size = DATA_SIZE - test_size - valid_size
    train_set, test_set, valid_set = split_data(data, train_size, test_size, valid_size)

    if verbose:
        stderr.write("Train set size: {}\n".format(train_size))
        stderr.write("Test set size: {}\n".format(test_size))
        stderr.write("Valid set size: {}\n".format(valid_size))


    write_data(train_set, TRAIN_SET_PATH)
    if verbose:
        stderr.write("Train set was dumped to the file {}\n".format(TRAIN_SET_PATH))
    write_data(test_set, TEST_SET_PATH)
    if verbose:
        stderr.write("Test set was dumped to the file {}\n".format(TEST_SET_PATH))
    write_data(valid_set, VALID_SET_PATH)
    if verbose:
        stderr.write("Valid set was dumped to the file {}\n".format(VALID_SET_PATH))

    c = 1.0 / 2 ** 10
    cv_ans = []
    for _ in range(40):
        if verbose:
            stdout.write("C = {}\n".format(c))

        b, w, _ = train_svm(train_set, c)
        valid_ans = test_svm(valid_set, b, w)
        results = calculate_results(valid_set, valid_ans)
        err_rate = error_rate(results)
        cv_ans.append((c, err_rate))
        c *= 2

    if verbose:
        stderr.write(str(cv_ans) + "\n")
    bestC = min(cv_ans, key = lambda p: p[1])[0]
    b, w, _ = train_svm(train_set, bestC)
    test_ans = test_svm(test_set, b, w)
    results = calculate_results(test_set, test_ans)

    err_rate = error_rate(results)
    prec = precision(results)
    rec = recall(results)
    f1 = f1score(results)
    #print("C = {}, error rate = {}".format(c, err_rate))

    return (b, w, err_rate, prec, rec, f1)

# random.seed(0) # uncomment to make the program deterministic

cnt = 10
serr = 0.0
sprec = 0.0
srec = 0.0
sf1 = 0.0

for i in range(cnt):
    stderr.write("Running... {}/{}\n".format(i, cnt))
    _, _, err_rate, prec, rec, f1 = run_all(data, verbose = False)
    stderr.write("Error rate: {}%\n".format(err_rate * 100))
    serr += err_rate
    sprec += prec
    srec += rec
    sf1 += f1

print("Average error rate ({} runs) is {}%".format(cnt, serr / cnt * 100))
print("Average precision ({} runs) is {}%".format(cnt, sprec / cnt * 100))
print("Average recall ({} runs) is {}%".format(cnt, srec / cnt * 100))
print("Average f1 score ({} runs) is {}".format(cnt, sf1 / cnt))