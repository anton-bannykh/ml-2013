from mlaux import *
import numpy as np
from scipy.optimize import minimize


def train(data, lam=0):
    x, y = split_xy(data)
    n, d = x.shape

    def f(theta):
        theta_0, theta_ = theta[-1], theta[:-1]
        logarifms = np.log(1 + np.exp(-y * (np.dot(x, theta_) + theta_0)))
        return float(lam) / 2 * np.sum(theta_ * theta_) + np.sum(logarifms)

    ans = minimize(f, np.zeros(d + 1)).x
    return ans


def avgerr(data, theta):
    x, y = split_xy(data)
    theta_0, theta_ = theta[-1], theta[:-1]
    preds = float(1) / (1 + np.exp(-y * (np.dot(x, theta_) + theta_0)))
    return np.average(1 - preds)


def opt_lambda(data):
    reg_best, err_best = 0, 1
    data_train, data_test = split(data)
    for d in range(-40, 10):
        lam = 2 ** d
        theta = train(data_train, lam=lam)
        err = avgerr(data_test, theta)
        if err_best > err:
            reg_best, f1_best = lam, err
    return reg_best


data = grouped(load_data())
data_train, data_test = split(data)
lam = opt_lambda(data)
theta = train(data_train, lam=lam)
err = avgerr(data_test, theta)

print("Average error: %.2f\n" % err)
print("Regularization constant used: %s\n" % str(lam))

