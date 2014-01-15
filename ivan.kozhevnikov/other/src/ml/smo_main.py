import util
from smo import Smo
import numpy as np
import sys


def kernel(x, y):
    a = np.array(x)
    b = np.array(y)
    return a.dot(b)


def builder(c, vectors, classes):
    return Smo(c, vectors, classes, 0.01, kernel)

util.run(builder, sys.argv[1])
