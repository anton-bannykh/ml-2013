__author__ = 'max'

from numpy import array
import numpy as np
from cvxopt.solvers import qp, options
from cvxopt import matrix


def unzip(tr):
    xs = []
    ys = []
    for (_, y, x) in tr:
        xs.append(x)
        ys.append(y)
    return xs, ys


def lin_kernel(x1, x2):
    return array(x1).dot(array(x2))

cached_params = None


def get_params(xs, ys, C):
    l = len(ys)
    global cached_params
    if cached_params is None:
        P = matrix(0., (l, l))
        for i in range(l):
            for j in range(l):
                P[i, j] = ys[i] * ys[j] * lin_kernel(xs[i], xs[j])

        q = matrix(-1., (l, 1))
        A = matrix(ys, (1, l), 'd')
        b = matrix(0., (1, 1))
        cached_params = (P, q, A, b)
    (P, q, A, b) = cached_params
    G = matrix(0., (2 * l, l))
    h = matrix(0., (2 * l, 1))
    for i in range(l):
        G[2 * i, i] = 1
        h[2 * i, 0] = C
        G[2 * i + 1, i] = -1

    return P, q, G, h, A, b


def train_svm(C, training):
    xs, ys = unzip(training)

    sol = qp(*get_params(xs, ys, C))
    lam = array(sol['x']).flat

    xa = array(xs).transpose()
    ya = array(ys)
    w = np.sum((lam * ya * xa), axis=1)
    w0 = np.median(w.dot(xa) - ya)
    return lambda x: array(x).dot(w) - w0


def split_ids(l, parts):
    sl = int(l / parts)
    return [(sl*i, sl*(i + 1)) for i in range(parts)]


def cross_validation(input_set):
    splits = split_ids(len(input_set), 5)
    best_C = None
    best_result = None

    for i in range(-5, 10, 1):
        C = 2.0 ** i
        print "Trying C =", C
        error_rates = []
        for (test_start, test_end) in splits:
            test = input_set[test_start:test_end]
            train = input_set[:test_start] + input_set[test_end:]
            svm = train_svm(C, train)
            errors = 0.
            for (_, y, x) in test:
                res = svm(x)
                if y * res <= 0:
                    errors += 1
            error_rates.append(errors / len(test))

        new_result = sum(error_rates)
        if best_result is None or new_result < best_result:
            best_result = new_result
            best_C = C
    print("best C = ", C)
    global cached_params
    cached_params = None
    return best_C


def start(training, test):
    options['show_progress'] = False
    C = cross_validation(training)
    svm = train_svm(C, training)

    tp = 0.
    fp = 0.
    fn = 0.
    for (_, y, x) in test:
        out = svm(x)
        if y > 0 and out > 0:
            tp += 1
        elif y > 0 > out:
            fn += 1
        elif y < 0 < out:
            fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print "Perceptron results"
    print "Precision: %4.2f%%" % (precision * 100.)
    print "Recall: %4.2f%%" % (recall * 100.)
    print "F1-metric: %4.2f%%" % (200. * precision * recall / (precision + recall))
