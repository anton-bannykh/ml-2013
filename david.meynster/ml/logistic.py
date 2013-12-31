import numpy as np
from scipy.optimize import minimize
from ml.data import split_xy

def train(data, l=0):
    x, y = split_xy(data)
    n, d = x.shape
    def f(theta):
        theta0, theta1 = theta[-1], theta[:-1]
        loglike = np.log(1 + np.exp(-y * (np.dot(x, theta1) + theta0)))
        return l / 2 * np.sum(theta1 * theta1) + np.sum(loglike)
    return minimize(f, np.zeros(d + 1)).x

def average_error(data, theta):
    x, y = split_xy(data)
    theta0, theta1 = theta[-1], theta[:-1]
    predictions = 1 / (1 + np.exp(-y * (np.dot(x, theta1) + theta0)))
    return np.average(1 - predictions)
