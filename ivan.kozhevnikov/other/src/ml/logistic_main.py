import util
import sys
from logistic import Logistic


def builder(c, vectors, classes):
    return Logistic(c, vectors, classes)


util.run(builder, sys.argv[1])