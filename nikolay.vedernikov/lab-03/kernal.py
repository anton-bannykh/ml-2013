import numpy as np
from math import exp

def scalar(x, y):
    return float(np.dot(x, y))

def polynomial(x, y, p=4):
    return float(1 + np.dot(x, y)) ** p

def gaussian(x, y, beta=5e-6):
    return exp(-np.sum((x - y) ** 2) * beta / 2)
