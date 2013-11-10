#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cvxopt import matrix
from cvxopt.solvers import qp

def scale(xs, c):
    return [x * c for x in xs]

def base_vector(i, n):
    return [1.0 if j == i else 0.0 for j in range(n)]

def get_qp_parameters(xs, ys, kernel, C):
    P = []
    q = []
    G = []
    h = []

    n = len(xs)
    for i, (x1, y1) in enumerate(zip(xs, ys)):
        q.append(-1.0)
        P_row = []
        for j, (x2, y2) in enumerate(zip(xs, ys)):
            P_row.append(-y1 * y2 * kernel(x1, x2))
        P.append(P_row)
        G.append(base_vector(i, n))
        h.append(C)

        G.append(scale(base_vector(i, n)))
        h.append(0)

    return [matrix(m) for m in [P, q, G, h]]

def main():
    pass

if __name__ == "__main__":
    main()