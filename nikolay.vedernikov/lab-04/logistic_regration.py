import numpy as np
from numpy.linalg import norm
import scipy.optimize as opt
import main

def train(data, lymbda=0):
    x, y = data[:, :-1], data[:, -1]
    n, d = x.shape
    def f(theta):
        theta0, theta1 = theta[-1], theta[:-1]
        return 0.5 * lymbda * norm(theta1) + sum(np.log(1. + np.exp(-y * (np.dot(theta1, x.T) + theta0))))
    return opt.minimize(f, np.zeros(d + 1)).x

def testing(data, theta):
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    result = {'pre': 0.0, 'rec': 0.0, 'er': 0.0, 'f1': 0.0}
    x, y = data[:, :-1], data[:, -1]
    theta0, theta1 = theta[-1], theta[:-1]
    a = np.dot(x, theta1) + theta0
    a[a < 0], a[a >= 0] = -1, 1
    for i in range(len(y)):
        yc = a[i]
        if yc == 1 and y[i] == 1:
            stats['tp'] += 1
        elif yc == 1 and y[i] == -1:
            stats['fp'] += 1
        elif yc == -1 and y[i] == 1:
            stats['fn'] += 1
        else:
            stats['tn'] += 1

    result['pre'] = stats['tp'] / (stats['tp'] + stats['fp'])
    result['rec'] = stats['tp'] / (stats['tp'] + stats['fn'])
    result['er'] = (stats['fp'] + stats['fn']) / len(y)
    result['f1'] = 2 * result['rec'] * result['pre'] / (result['rec'] + result['pre'])
    return result

def optimize_regularization(data):
    lymbda_best, f1_best = 0, 0
    data_train, data_test = main.split(data, 0.5)
    for d in range(-7, 10):
        lymbda = 2 ** d
        theta = train(data_train, lymbda=lymbda)
        res = testing(data_test, theta)
        f1_cur = res['f1']
        print("Current lymbda: %.5f" % lymbda)
        if f1_best < f1_cur:
            lymbda_best, f1_best = lymbda, f1_cur
            print("Now best F1: %.5f" % f1_best)

    return lymbda_best
