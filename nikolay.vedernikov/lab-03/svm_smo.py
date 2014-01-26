import numpy as np
import random as rand
from random import randint
from kernal import scalar
from kernal import polynomial

rand.seed("canser")

import main

def classify(alpha, b, x, y, v, kernel=scalar):
    return b + np.sum(alpha * y * np.apply_along_axis(lambda w: kernel(v, w), 1, x))

def train(data, c, kernel=scalar, tol=1e-9):
    def crop(p, lo, hi):
        return lo if p < lo else (hi if p > hi else p)

    x, y = data[:, :-1], data[:, -1]
    n, d = x.shape

    alpha, b = np.zeros(n), 0

    k = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            k[i][j] = kernel(x[i], x[j])
    ff = lambda i: b + np.sum(np.dot(k[i], alpha * y))
    changed = True
    while changed:
        changed = False
        for i in range(n):
            Ei = ff(i) - y[i]
            if (y[i] * Ei < -tol and alpha[i] < c) or (y[i] * Ei > tol and alpha[i] > 0):
                j = randint(0, n - 2)
                if j == i:
                    j += 1
                Ej = ff(j) - y[j]
                ai, aj = alpha[i], alpha[j]
                if y[i] != y[j]:
                    L, H = max(0, aj - ai), min(c, c + aj - ai)
                else:
                    L, H = max(0, ai + aj - c), min(c, ai + aj)

                if L >= H:
                    continue
                eta = 2 * k[i][j] - k[i][i] - k[j][j]
                if eta > 0:
                    continue
                alpha[j] = crop(alpha[j] - y[j] * (Ei - Ej) / eta, L, H)
                if abs(alpha[j] - aj) < 1e-5:
                    continue

                alpha[i] = alpha[i] + y[i] * y[j] * (aj - alpha[j])
                b1 = b - Ei - y[i] * (alpha[i] - ai) * kernel(x[i], x[i]) - y[j] * (alpha[j] - aj) * kernel(x[i], x[j])
                b2 = b - Ej - y[i] * (alpha[i] - ai) * kernel(x[i], x[j]) - y[j] * (alpha[j] - aj) * kernel(x[j], x[j])
                if 0 < alpha[i] < c:
                    b = b1
                elif 0 < alpha[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                changed = True

    return alpha, b

def testing(data_train, test, p, kernel=scalar):
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    result = {'pre': 0.0, 'rec': 0.0, 'er': 0.0, 'f1': 0.0}
    x, y = data_train[:, :-1], data_train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    alpha, b = p

    for i in range(len(y_test)):
        yc = classify(alpha, b, x, y, x_test[i], kernel)
        if yc >= 0 and y_test[i] == 1:
            stats['tp'] += 1
        elif yc >= 0 and y_test[i] == -1:
            stats['fp'] += 1
        elif yc < 0 and y_test[i] == 1:
            stats['fn'] += 1
        else:
            stats['tn'] += 1

    result['pre'] = stats['tp'] / (stats['tp'] + stats['fp'])
    result['rec'] = stats['tp'] / (stats['tp'] + stats['fn'])
    result['er'] = (stats['fp'] + stats['fn']) / len(y_test)
    result['f1'] = 2 * result['rec'] * result['pre'] / (result['rec'] + result['pre'])
    return result

