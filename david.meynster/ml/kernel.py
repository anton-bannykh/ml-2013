import numpy as np
from math import exp

def gaussian(x, y, beta=1e-6):
    return exp(-np.sum((x - y) ** 2) * beta / 2)

def polynomial(x, y, p=3):
    return float(1 + np.dot(x, y)) ** p

def scalar(x, y):
    return float(np.dot(x, y))