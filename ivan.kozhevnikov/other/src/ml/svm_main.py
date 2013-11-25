import util
from svm import Svm
import sys


def builder(c, vectors, classes):
    return Svm(c, vectors, classes)


util.run(builder, sys.argv[1])