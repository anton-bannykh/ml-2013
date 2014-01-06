import random

import numpy as np

from common.Reader import *


def classify(w, x):
    return 1 if np.inner(w, x) > 0 else -1

def linear_regression(x, y):
    return np.dot(np.linalg.pinv(x), y)

def misclassified(samples, w):
    return [(x, y) for (x, y) in samples if (classify(w, x) != y)]

def train_perceptron(x, y, iterations=1000):
    w = linear_regression(x, y)
    samples = list(zip(x, y))

    cur_misclassified = misclassified(samples, w)
    best_w, best_result = list(w), len(cur_misclassified)

    for _ in range(iterations):
        if (best_result == 0):
            break

        xCur, yCur = random.choice(cur_misclassified)
        w += yCur * xCur

        cur_misclassified = misclassified(samples, w)
        if best_result > len(cur_misclassified):
            best_w = w.copy()
            best_result = len(cur_misclassified)

    return best_w, best_result / len(x)

x, y = get_data()
size = int(len(x) * 0.1)

w, Ein = train_perceptron(x[size:], y[size:])

result = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

for xCur, yCur in zip(x[:size], y[:size]):
    yc = classify(w, xCur)
    if yCur == -1:
        key = 'tn' if yc == -1 else 'fp'
        result[key] += 1
    if yCur == 1:
        key = 'tp' if yc == 1 else 'fn'
        result[key] += 1

precision = result['tp'] / (result['tp'] + result['fp'])
recall = result['tp'] / (result['tp'] + result['fn'])
F1 = 2 * precision * recall / (precision + recall)
Eout = (result['fp'] + result['fn']) / len(x[:size])

print('in sample error     = %6.2f' % (100 * Ein))
print('out of sample error = %6.2f' % (100 * Eout))
print('precision           = %6.2f' % (100 * precision))
print('recall              = %6.2f' % (100 * recall))
print('F1                  = %6.2f' % F1)