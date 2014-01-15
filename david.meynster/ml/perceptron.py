import numpy as np
from ml.stats import Stats
from ml.data import split_xy

def classify(theta, x):
    return 1 if np.dot(theta, x) >= 0 else -1

def train(data, iterations=1000):
    x, y = split_xy(data)
    n, d = x.shape
    theta = np.zeros(d)

    for it in range(iterations):
        for i in range(n):
            if classify(theta, x[i]) != y[i]:
                theta += y[i] * x[i]
    return theta

def test(data, theta):
    x, y = split_xy(data)
    stats = Stats()
    for i in range(len(y)):
        yc = classify(theta, x[i])
        if yc == 1 and y[i] == 1:
            stats.tp += 1
        elif yc == 1 and y[i] == -1:
            stats.fp += 1
        elif yc == -1 and y[i] == 1:
            stats.fn += 1
        else:
            stats.tn += 1
    return stats