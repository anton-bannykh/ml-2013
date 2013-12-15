import util
from network import NeuronClassifier
import sys

def builder(c, vectors, classes):
    return NeuronClassifier(c, vectors, classes)


util.run(builder, sys.argv[1])

