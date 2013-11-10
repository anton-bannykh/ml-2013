#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from util import split_data, append, average, unzip
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
        raise Exception("Haven't found any margin vectors, which is weird")

    return w, b

FOLD_K = 5

def classify(x, w, b):
    if np.dot(w, x) + b > 0:
        return 1
    else:
        return -1

def svm_crossvalidate(xs, ys, C, kernel):
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

def main():
    options['show_progress'] = False
    xs, ys = retrieve_data(as_list=True, negative_label=-1.0, positive_label=1.0)
    for i in range(-3, 3):
        C = 10.0**i
        print(i, svm_crossvalidate(xs, ys, C, linear_kernel))

if __name__ == "__main__":
    main()