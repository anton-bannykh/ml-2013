from ml.stats import Stats
from ml.data import split_xy
import numpy as np
from random import randint

def train(data, layers=2, alpha=0.1, lsize=100, iterations=5000):
    x, y = split_xy(data)
    n, d = x.shape
    y = (1 + y) / 2

    ns = [d] + [lsize] * layers + [1]
    weights = []
    for i in range(layers + 1):
        weights.append(np.random.random((ns[i + 1], ns[i])) - 0.5)

    def forward(x):
        g = lambda x: 1 / (1 + np.exp(-x))
        a, z = [x], [x]
        for w in weights:
            a.append((w.dot(z[-1])))
            z.append(g(a[-1]))
        return a, z

    def backward(a, z, y):
        dl = z[-1] - y
        for i in range(layers, -1, -1):
            dj = z[i] * (1 - z[i]) * np.dot(weights[i].transpose(), dl)
            weights[i] -= alpha * np.outer(dl, a[i])
            dl = dj

    for _ in range(iterations):
        i = randint(0, n - 1)
        a, z = forward(x[i])
        backward(a, z, y[i])

    return lambda x: forward(x)[1][-1][0]


def test(data, network):
    x, y = split_xy(data)
    stats = Stats()
    for i in range(len(y)):
        yc = 1 if network(x[i]) > 0.5 else -1
        if yc == 1 and y[i] == 1:
            stats.tp += 1
        elif yc == 1 and y[i] == -1:
            stats.fp += 1
        elif yc == -1 and y[i] == 1:
            stats.fn += 1
        else:
            stats.tn += 1
    return stats