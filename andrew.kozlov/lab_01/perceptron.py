from random import choice
import numpy as np

__author__ = 'adkozlov'


def initial_weights(data):
    xs = [x for (x, _) in data]
    ys = [y for (_, y) in data]

    return np.dot(np.linalg.pinv(xs), ys)


def classify(w, x):
    return 1 if np.inner(w, x) > 0 else -1


def equal(x, y, eps=1e-3):
    return abs(x - y) < eps


def misclassified(data, w):
    return [(x, y) for (x, y) in data if not equal(classify(w, x), y)]


def train(data, n=1000):
    w = initial_weights(data)

    best_w, best_result = None, 0
    for _ in range(n):
        mc = misclassified(data, w)

        if best_w is None or best_result > len(mc):
            best_w = w.copy()
            best_result = len(mc)
        if best_result == 0:
            break

        x, y = choice(mc)
        if y == 1:
            w += y * x
        else:
            w -= -y * x

    return best_w