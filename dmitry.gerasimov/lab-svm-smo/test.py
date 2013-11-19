import os
from sys import stdin, stdout, stderr

import numpy as np

from bcwd import *
from svm_smo import *
from common import *
import kernels

# linearly separable set
def test1():
    data = [Entry(id = None, correct = 1, features = np.array([2.0, 5.0])),
            Entry(id = None, correct = 1, features = np.array([4.0, 5.0])),
            Entry(id = None, correct = 1, features = np.array([5.0, 7.0])),
            Entry(id = None, correct = -1, features = np.array([5.0, 1.0])),
            Entry(id = None, correct = -1, features = np.array([6.0, 3.0])),
            Entry(id = None, correct = -1, features = np.array([8.0, 3.0]))]
    b, w = train_svm(data, 1000000.0, kernels.identity)
    print("b = {}, w = {}".format(b, w))
    test_ans = test_svm(data, b, w)
    results = calculate_results(data, test_ans)
    print(results)


def test2():
    data = [Entry(id = None, correct = -1, features = np.array([1.0])),
            Entry(id = None, correct = -1, features = np.array([2.0])),
            Entry(id = None, correct = 1, features = np.array([3.0])),
            #Entry(id = None, correct = -1, features = np.array([4.0])),
            Entry(id = None, correct = 1, features = np.array([5.0])),
            Entry(id = None, correct = 1, features = np.array([6.0])),
            Entry(id = None, correct = 1, features = np.array([7.0])),]
    b, w = train_svm(data, 100.0, kernels.identity)
    print("b = {}, w = {}".format(b, w))
    test_ans = test_svm(data, b, w)
    results = calculate_results(data, test_ans)
    print(results)

test2()