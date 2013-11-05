import numpy as np
import math

def classify(theta, x):
    return 1 if np.dot(theta, x) >= 0 else -1

def train(x, y, iterations=1000):
    n, d = len(y), len(x[0])
    theta = np.zeros(d)

    for it in range(iterations):
        for i in range(n):
            if classify(theta, x[i]) != y[i]:
                theta += y[i] * x[i]
    return theta

def test(x, y, theta):
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for i in range(len(x)):
        yc = classify(theta, x[i])
        if yc == 1 and y[i] == 1:
            stats['tp'] += 1
        elif yc == 1 and y[i] == -1:
            stats['fp'] += 1
        elif yc == -1 and y[i] == 1:
            stats['fn'] += 1
        else:
            stats['tn'] += 1
    return stats