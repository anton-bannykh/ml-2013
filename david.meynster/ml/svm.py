import numpy as np
from scipy.optimize import minimize
from ml.stats import Stats
from ml.data import split_xy

def train(data, c=0):
    x, y = split_xy(data)
    n, d = x.shape
    def f(theta):
        theta0, theta1 = theta[-1], theta[:-1]
        pred = y * (np.dot(x, theta1) + theta0)
        return 0.5 * np.sum(theta1 * theta1) + c * np.sum(1 - pred[pred < 1])
    return minimize(f, np.zeros(d + 1)).x

def test(data, theta):
    x, y = split_xy(data)
    stats = Stats()
    theta0, theta1 = theta[-1], theta[:-1]
    pred = np.dot(x, theta1) + theta0
    pred[pred < 0], pred[pred >= 0] = -1, 1
    for i in range(len(y)):
        if pred[i] == 1 and y[i] == 1:
            stats.tp += 1
        elif pred[i] == 1 and y[i] == -1:
            stats.fp += 1
        elif pred[i] == -1 and y[i] == 1:
            stats.fn += 1
        else:
            stats.tn += 1
    return stats

