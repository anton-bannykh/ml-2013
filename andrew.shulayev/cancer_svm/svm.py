#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from util import *
from cvxopt import matrix
from cvxopt.blas import dotu
from cvxopt.solvers import qp, options
from cancer_common.data import retrieve_data
import numpy as np

def scale(xs, c):
    return [x * c for x in xs]

def base_vector(i, n):
    return [1.0 if j == i else 0.0 for j in range(n)]

def transpose(ls):
    return list(map(list, zip(*ls)))

def get_qp_parameters(xs, ys, kernel, C):
    n = len(xs)

    P = []
    q = []
    G = []
    h = []
    A = [ys]
    b = [0.0]

    for i, (x1, y1) in enumerate(zip(xs, ys)):
        q.append(-1.0)
        P_row = []
        for x2, y2 in zip(xs, ys):
            P_row.append(y1 * y2 * kernel(x1, x2))
        P.append(P_row)
        G.append(base_vector(i, n))
        h.append(C)

        G.append(scale(base_vector(i, n), -1.0))
        h.append(0.0)

    return [matrix(m) for m in [P, q, transpose(G), [h], transpose(A), b]]

def linear_kernel(xs, ys):
    return dotu(matrix(xs), matrix(ys))

def dump(*args):
    for x in args:
        print(x)
        print()

class SVMException(Exception):
    pass

def train_svm(xs, ys, C, kernel):
    qp_solution = qp(*get_qp_parameters(xs, ys, kernel, C))

    eps = 1e-5

    a = np.array(qp_solution['x']).transpose()[0]
    w = np.zeros(len(xs[0]))
    for ai, xi, yi in zip(list(a), xs, ys):
        if ai < eps:
            continue
        w += yi * ai * np.array(xi)

    b = None
    for i, ai in enumerate(a):
        if eps < ai < (C - eps):
            b = ys[i] - np.dot(w, xs[i])
            break

    if b is None:
        raise SVMException("Haven't found any margin vectors, is system consistent?")

    return w, b

FOLD_K = 5

def classify(x, w, b):
    if np.dot(w, x) + b > 0:
        return 1
    else:
        return -1

def svm_cross_validate(xs, ys, C, kernel):
    parts = split_data(xs, ys, FOLD_K)
    errors = []
    for i, test_data in enumerate(parts):
        train_data = append(parts[:i] + parts[i+1:])
        xs, ys = unzip(*train_data)

        w, b = train_svm(xs, ys, C, kernel)
        misclassified = 0
        for x, y in test_data:
            if classify(x, w, b) != int(y):
                misclassified += 1
        errors.append(misclassified / len(test_data))

    return average(errors)

def optimal_regularizer(xs, ys, kernel):
    best_C, best_result = None, None
    for i in range(-4, 6):
        C = 10.0**i
        try:
            result = svm_cross_validate(xs, ys, C, kernel)
            print('C = %4.1e, error = %5.3f' % (C, result))
            if best_result is None or result < best_result:
                best_result = result
                best_C = C
        except SVMException as e:
            print('C = %4.1e, got exception: %s' % (C, e))
    return best_C

def main(train_ratio=0.9):
    options['show_progress'] = False
    xs, ys = shuffle_args(*retrieve_data(as_list=True, negative_label=-1.0, positive_label=1.0))
    xs_train, xs_test = split_with_ratio(xs, train_ratio)
    ys_train, ys_test = split_with_ratio(ys, train_ratio)

    C = optimal_regularizer(xs_train, ys_train, linear_kernel)
    print('best C=%4.1e' % C)
    w, b = train_svm(xs_train, ys_train, C, linear_kernel)

    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for x, y in zip(xs_test, ys_test):
        yc = classify(x, w, b)
        if int(y) == -1:
            key = 'tn' if yc == -1 else 'fp'
            stats[key] += 1
        if int(y) == 1:
            key = 'tp' if yc == 1 else 'fn'
            stats[key] += 1

    precision = safe_division(stats['tp'], stats['tp'] + stats['fp'])
    recall = safe_division(stats['tp'], stats['tp'] + stats['fn'])
    f1 = safe_division(2 * precision * recall, precision + recall)

    print('precision  = %6.2f%%' % (100 * precision))
    print('recall     = %6.2f%%' % (100 * recall))
    print('F1 measure = %6.2f' % f1)

if __name__ == "__main__":
    main()