import numpy as np
from numpy.linalg import norm
import scipy.optimize as opt


class Logistic:
    def __init__(self, alpha, xs, ys):
        def f(theta):
            theta0, theta1 = theta[-1], theta[:-1]
            theta_ar = np.array(theta1)
            return alpha * norm(theta1) / 2 + sum([np.log(1 + np.exp(-np.array(x).dot(theta_ar) * y)) for x, y in zip(xs, ys)])

        self.theta = opt.minimize(f, np.zeros(len(xs[0]) + 1)).x

    def classify(self, x):
        theta0, theta1 = self.theta[-1], self.theta[:-1]
        res = np.dot(theta1, x) + theta0
        if res > 0:
            return 1
        else:
            return -1

