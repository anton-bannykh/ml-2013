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

TEST_SET_FRACTION = 0.10

TRAINING_ITERATIONS = 100

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

#get_data(DATA_URL, DATA_LOCAL_PATH)
#stderr.write("Data set fetched to the file {}\n".format(DATA_LOCAL_PATH))
#stderr.write("Comment the line 35 of main.py to prevent downloading during the further runs\n")

data = load_data(DATA_LOCAL_PATH)
assert len(data) == DATA_SIZE # there should be 569 instances according to the set description

from cvxopt import matrix, solvers

training_set = data
m = len(training_set) # number of trining examples
dim = 30 # dimension of the feature vector # TODO BIASED!!!
C = 1.0 # regularization constant


pP = [[0.0 for _ in range(1 + dim + m)] for _ in range(1 + dim + m)]
# set ||w||^2 constraint
for i in range(1, 1 + dim):
    pP[i][i] = 1.0
P = matrix(pP)

pq = [0.0 for _ in range(1 + dim + m)]
for i in range(1 + dim, 1 + dim + m):
    pq[i] = C
q = matrix(pq)


pG = [[0.0 for _ in range(1 + dim + m)] for _ in range(2 * m)]
# set \xi_i >= 0 constraint
for i in range(0, m):
    pG[i][1 + dim + i] = -1.0
# set constraints on training set points
for i in range(0, m):
    point = training_set[i]
    line = pG[i + m]
    # bias coefficient
    line[0] = -point.correct
    # dot product coeffecients
    for j, f in enumerate(point.features):
        line[1 + j] = -point.correct * f
    # regularisation coeffecient
    line[1 + dim + i] = -1.0
G = matrix(([[pG[i][j] for i in range(2 * m)] for j in range(1 + dim + m)])) # cvxopt uses column-major order

ph = [0.0 for i in range(2 * m)]
# \xi_i >= 0, no coefficients on the rhs
# coefficients for training set points constraints:
for i in range(0, m):
    ph[m + i] = -1.0
h = matrix(ph)

# no equality constraints

sol = solvers.qp(P, q, G, h)
print(sol['x'])