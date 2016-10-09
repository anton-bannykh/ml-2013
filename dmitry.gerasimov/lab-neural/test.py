import os
from sys import stdin, stdout, stderr

import numpy as np

from bcwd import *
from neural_network import *
from common import *

def test111():
    # learning function [x > 3]
    #data = [Entry(id = None, correct = 1, features = np.array([6.0])),
    #        Entry(id = None, correct = 1, features = np.array([6.0])),
    #        Entry(id = None, correct = -1, features = np.array([0.0])),
    #        Entry(id = None, correct = -1, features = np.array([0.0]))
    #        ]
    #data = [Entry(id = None, correct = 1, features = np.array([4.0])),
    #        Entry(id = None, correct = 1, features = np.array([5.0])),
    #        Entry(id = None, correct = 1, features = np.array([6.0])),
    #        Entry(id = None, correct = 1, features = np.array([7.0])),
    #        Entry(id = None, correct = 1, features = np.array([8.0])),
    #        Entry(id = None, correct = 1, features = np.array([9.0])),
    #        Entry(id = None, correct = -1, features = np.array([0.0])),
    #        Entry(id = None, correct = -1, features = np.array([1.0])),
    #        Entry(id = None, correct = -1, features = np.array([-1.0])),
    #        Entry(id = None, correct = -1, features = np.array([-4.0])),
    #        ]
    classifier = train_neural_network_2(data)
    test_ans = test_neural_network(data, classifier)
    results = calculate_results(data, test_ans)
    print(results)

# linearly separable set
def test1():
    # 1: x < y
    # 2 : x > y
    data = [Entry(id = None, correct = 1, features = np.array([4.0, 5.0])),
            Entry(id = None, correct = 1, features = np.array([6.0, 5.0])),
            Entry(id = None, correct = 1, features = np.array([7.0, 7.0])),
            Entry(id = None, correct = 1, features = np.array([6.0, 7.0])),
            Entry(id = None, correct = 1, features = np.array([5.0, 5.0])),
            Entry(id = None, correct = 1, features = np.array([4.0, 6.0])),
            Entry(id = None, correct = -1, features = np.array([7.0, 1.0])),
            Entry(id = None, correct = -1, features = np.array([8.0, 3.0])),
            Entry(id = None, correct = -1, features = np.array([10.0, 3.0])),
            Entry(id = None, correct = -1, features = np.array([9.0, 5.0])),
            Entry(id = None, correct = -1, features = np.array([8.0, 1.0])),
            Entry(id = None, correct = -1, features = np.array([7.0, 3.0])),
            ]
    classifier = train_neural_network_2(data)
    test_ans = test_neural_network(data, classifier)
    results = calculate_results(data, test_ans)
    print(results)

# linearly separable set
#def test2():
#    # 1: x < y
#    # 2 : x > y
#    data = [Entry(id = None, correct = 1, features = np.array([0.35, 0.9]))]
#    classifier = train_neural_network_3(data)
#    test_ans = test_neural_network(data, classifier)
#    results = calculate_results(data, test_ans)
#    print(results)


#
#def test2():
#    data = [Entry(id = None, correct = -1, features = np.array([1.0])),
#            Entry(id = None, correct = -1, features = np.array([2.0])),
#            Entry(id = None, correct = 1, features = np.array([3.0])),
#            Entry(id = None, correct = -1, features = np.array([4.0])),
#            Entry(id = None, correct = 1, features = np.array([5.0])),
#            Entry(id = None, correct = 1, features = np.array([6.0])),
#            Entry(id = None, correct = 1, features = np.array([7.0])),]
#    classifier = train_svm(data, 100.0, kernels.identity)
#    test_ans = test_svm(data, classifier)
#    results = calculate_results(data, test_ans)
#    print(results)
#
#def test3():
#    data = [Entry(id = None, correct = -1, features = np.array([1.0, 1.0])),
#            Entry(id = None, correct = -1, features = np.array([-1.0, 1.0])),
#            Entry(id = None, correct = -1, features = np.array([1.0, -1.0])),
#            Entry(id = None, correct = -1, features = np.array([-1.0, -1.0])),
#            Entry(id = None, correct = 1, features = np.array([5.0, 5.0])),
#            Entry(id = None, correct = 1, features = np.array([-5.0, 5.0])),
#            Entry(id = None, correct = 1, features = np.array([5.0, -5.0])),
#            Entry(id = None, correct = 1, features = np.array([-5.0, -5.0]))]
#
#    classifier = train_svm(data, 100.0, lambda x, y: kernels.poly2(x, y, 0.0))
#    test_ans = test_svm(data, classifier)
#    results = calculate_results(data, test_ans)
#    print(results)
#
#
#test3()
test1()