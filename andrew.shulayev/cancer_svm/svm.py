#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cvxopt import matrix
from cvxopt.blas import dotu
from cvxopt.solvers import qp
import numpy as np

def scale(xs, c):
    return [x * c for x in xs]

def base_vector(i, n):
    return [1.0 if j == i else 0.0 for j in range(n)]

def transpose(ls):
    result = list(map(list, zip(*ls)))
    print(result)
    return result

def get_qp_parameters(xs, ys, kernel, C):
    n = len(xs)

    P = []
    q = []
    G = [ys]
    h = [0]

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

    return [matrix(m) for m in [P, q, transpose(G), [h]]]

def linear_kernel(xs, ys):
    print(xs, ys)
    return dotu(matrix(xs), matrix(ys))

def dump(*args):
    for x in args:
        print(x)
        print()

def main():
    xs = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    ys = [-1.0, 1.0, 1.0]
    solution = qp(*get_qp_parameters(xs, ys, linear_kernel, 0.8))

    for a in solution['x']:
        print(a)

if __name__ == "__main__":
    main()