import numpy as np
import numpy.linalg as linalg
from math import *

def identity(va, vb):
    return np.dot(va, vb)

def poly(va, vb, c, p):
    return (np.dot(va, vb) + c) ** p

poly2 = lambda va, vb, c: poly(va, vb, c, 2)

def gaussian(va, vb, gamma):
    #print("Norm {}".format(linalg.norm(va - vb)))
    return exp(gamma * linalg.norm(va - vb) ** 2)