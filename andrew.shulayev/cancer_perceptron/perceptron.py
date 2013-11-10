#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from random import choice

def approx_equal(f1, f2):
    return abs(f1 - f2) < 0.001

def initial_weights(xs, ys):
    return np.dot(np.linalg.pinv(xs), ys)

def classify(w, x):
    prod = np.inner(w, x)
    if prod > 0:
        return 1
    else:
        return -1

def train_perceptron(xs, ys, iters=1000):
    w = initial_weights(xs, ys)
    samples = list(zip(xs, ys))

    best_w, best_result = None, None
    for _ in range(iters):
        misclassified = [(x, y) for (x, y) in samples if not approx_equal(classify(w, x), y)]
        if best_w is None or best_result > len(misclassified):
            best_w = w.copy()
            best_result = len(misclassified)
        if best_result == 0:
            break

        x, y = choice(misclassified)
        w += y * x

    return best_w
