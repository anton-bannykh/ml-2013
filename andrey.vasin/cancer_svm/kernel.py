import numpy
import math

def inner_product_kernel(x1, x2):
    return numpy.inner(x1, x2)

def polynomial_kernel(x1, x2, q = 5):
    return (1 + numpy.inner(x1, x2)) ** q

def gaussian_kernel(x1, x2, gamma = 1.0):
    diff = numpy.array(x1) - numpy.array(x2)
    norm_sq = numpy.inner(diff, diff)
    return math.exp(-gamma * norm_sq)