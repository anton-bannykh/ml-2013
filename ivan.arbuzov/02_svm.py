from mlaux import *
import numpy as np
from scipy.optimize import minimize


def train(data, C=0):
    x, y = split_xy(data)
    n, d = x.shape

    def f(theta):
        theta_0, theta_ = theta[-1], theta[:-1]
        pred = y * (np.dot(x, theta_) + theta_0)
        return 0.5 * np.sum(theta_ * theta_) + C * np.sum(1 - pred[pred < 1])

    ans = minimize(f, np.zeros(d + 1)).x
    return ans


def test(data, theta):
    x, y = split_xy(data)
    theta_0, theta_1 = theta[-1], theta[:-1]
    pred = np.dot(x, theta_1) + theta_0
    pred[pred < 0], pred[pred >= 0] = -1, 1
    return Statistic(pred, y)


def opt_reg_const(data):
    reg_best, f1_best = 0, 0
    data_train, data_test = split(data)
    for d in range(-10, 40):
        reg_current = 0.5 ** d
        theta = train(data_train, C=reg_current)
        stats = test(data_test, theta)
        f1_current = stats.f1()
        if f1_best < f1_current:
            reg_best, f1_best = reg_current, f1_current
    return reg_best

data = grouped(load_data())
data_train, data_test = split(data)
reg_const = opt_reg_const(data)
theta = train(data_train, C=reg_const)
stats = test(data_test, theta)

print("Precision: %.2f\nError: %.2f\nRecall: %.2f\nF1: %.2f\n" %
      (stats.precision(), stats.error(), stats.recall(), stats.f1()))
print("Regularization constant used: %.2f\n" % reg_const)

