__author__ = 'max'

from lab1.perceptron import *


def start(training, test):
    dim = len(training[0][2])
    p = get_untraining_perceptron(dim)
    p.train(0.05, training, 500)

    print(p.weights)

    tp = 0.
    fp = 0.
    fn = 0.
    for x in test:
        t = x[1]
        out = p.calc_output(x[2])
        if t > 0 and out > 0:
            tp += 1
        elif t > 0 > out:
            fn += 1
        elif t < 0 < out:
            fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print "Perceptron results"
    print "Precision: %4.2f%%" % (precision * 100.)
    print "Recall: %4.2f%%" % (recall * 100.)
    print "F1-metric: %4.2f%%" % (200. * precision * recall / (precision + recall))

