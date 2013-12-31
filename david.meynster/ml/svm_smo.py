import numpy as np
from ml.stats import Stats
from ml.data import split_xy
from ml.kernel import scalar
from random import randint

def classify(alpha, b, x, y, v, kernel=scalar):
    return b + np.sum(alpha * y * np.apply_along_axis(lambda w: kernel(v, w), 1, x))

def train(data, c, kernel=scalar, tol=1e-9):
    def crop(p, lo, hi):
        return lo if p < lo else (hi if p > hi else p)

    x, y = split_xy(data)
    n, d = x.shape
    alpha, b = np.zeros(n), 0

    # f = lambda v: classify(alpha, b, x, y, v, kernel)
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
                if abs(alpha[j] - aj) < 1e-5:  # lol why not tol? TODO: think about it
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

def test(data_train, data, p, kernel=scalar):
    x, y = split_xy(data_train)
    x_test, y_test = split_xy(data)
    n, n_test = len(y), len(y_test)
    alpha, b = p
    stats = Stats()
    f = lambda v: classify(alpha, b, x, y, v, kernel)

    for i in range(n_test):
        yc = f(x_test[i])
        if yc > 0 and y_test[i] == 1:
            stats.tp += 1
        elif yc > 0 and y_test[i] == -1:
            stats.fp += 1
        elif yc < 0 and y_test[i] == 1:
            stats.fn += 1
        else:
            stats.tn += 1

    return stats

