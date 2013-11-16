#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

from util import *
from random import randrange
from cancer_common.data import retrieve_data
import numpy as np

def bound(L, H, x):
    if x <= L:
        return L
    if x >= H:
        return H
    return x

def float_equals(x, y, tolerance=1e-5):
    return abs(x - y) < tolerance

def optimize_pair(t1, t2, C, f, b, kernel, tolerance=1e-5):
    a1, x1, y1 = t1
    a2, x2, y2 = t2

    # define clipping bounds
    if y1 != y2:
        L = max(0, a2 - a1)
        H = C + min(0, a2 - a1)
    else:
        L = max(0, a1 + a2 - C)
        H = min(C, a1 + a2)

    if float_equals(L, H, tolerance):
        return None

    E1 = f(x1) - y1
    E2 = f(x2) - y2
    n = 2 * kernel(x1, x2) - kernel(x1, x1) - kernel(x2, x2)

    if n >= 0:
        return None

    r2 = bound(L, H, a2 - y2 * (E1 - E2) / n)
    if float_equals(a2, r2, tolerance):
        return None

    r1 = a1 + y1 * y2 * (a2 - r2)

    b1 = b - E1 - y1 * (r1 - a1) * kernel(x1, x1) - y2 * (r2 - a2) * kernel(x1, x2)
    b2 = b - E2 - y1 * (r1 - a1) * kernel(x1, x2) - y2 * (r2 - a2) * kernel(x2, x2)

    if tolerance < a1 < C - tolerance:
        rb = b1
    elif tolerance < a2 < C - tolerance:
        rb = b2
    else:
        rb = (b1 + b2) * 0.5

    return r1, r2, rb

def target_function(a, xs, ys, b, kernel, x):
    result = 0
    for ai, xi, yi in zip(a, xs, ys):
        result += ai * yi * kernel(xi, x)
    return result + b

def sequential_minimal_optimization(xs, ys, C, kernel, max_passes=10, tolerance=1e-5):
    passes = 0
    n = len(xs)
    a = np.zeros(n)
    b = 0
    while passes < max_passes:
        changed = 0
        for i, (x, y) in enumerate(zip(xs, ys)):
            E = target_function(a, xs, ys, b, kernel, x) - y
            if not ((y * E < -tolerance and a[i] < C) or (y * E > tolerance and a[i] > C)):
                continue
            # KKT conditions are violated
            j = randrange(0, n - 1)
            if j >= i:
                j += 1
            f = lambda x: target_function(a, xs, ys, b, kernel, x)
            result = optimize_pair((a[i], x, y), (a[j], xs[j], ys[j]), C, f, b, kernel, tolerance)

            if result is None:
                # can't optimize this pair
                continue
            a1, a2, b = result
            a[i] = a1
            a[j] = a2
            changed += 1
        if changed == 0:
            passes += 1
        else:
            passes = 0
    return a, b

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(p, x, y):
    return (1 + np.dot(x, y))**p

def poly_kernel_wrapper(p):
    return lambda x, y: polynomial_kernel(p, x, y)

def gaussian_kernel(gamma, x, y):
    diff = np.array(x) - np.array(y)
    norm_sq = np.dot(diff, diff)
    return math.exp(-gamma * norm_sq)

def gaussian_kernel_wrapper(gamma):
    return lambda x, y: gaussian_kernel(gamma, x, y)

def check_kernel(train, test, C, kernel, name='unknown'):
    xs, ys = train
    a, b = sequential_minimal_optimization(xs, ys, C, kernel)

    xs, ys = test
    size = len(xs)
    errors = 0

    for x, y in zip(xs, ys):
        yc = sign(target_function(a, xs, ys, b, kernel, x))
        if yc != y:
            errors += 1
    rate = errors / size
    print('error rate on kernel "%s" = %8.3f%%' % (name, rate))

def main():
    xs, ys = retrieve_data(as_list=True)
    print('debug: data retrieved')
    xs_train, xs_test = split_with_ratio(xs, 0.9)
    ys_train, ys_test = split_with_ratio(ys, 0.9)

    train = (xs_train, ys_train)
    test = (xs_test, ys_test)
    C = 1e2

    check_kernel(train, test, C, linear_kernel, 'linear')
    for i in range(2, 5):
        check_kernel(train, test, C, poly_kernel_wrapper(i), 'polynomial %d' % i)
    check_kernel(train, test, C, gaussian_kernel_wrapper(1.0), 'gaussian')

if __name__ == "__main__":
    main()