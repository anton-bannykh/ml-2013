#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cancer_common.data import retrieve_data
from cancer_perceptron.perceptron import *
from util import safe_division

def main(test_fraction):
    xs, ys = retrieve_data()
    test_size = int(len(xs) * test_fraction)

    w = train_perceptron(xs[:-test_size], ys[:-test_size], iters=10000)

    # 'tp' — true positive
    # 'tn' — true negative
    # 'fp' — false positive
    # 'fn' — false negative
    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for x, y in zip(xs[-test_size:], ys[-test_size:]):
        yc = classify(w, x)
        if y == -1:
            key = 'tn' if yc == -1 else 'fp'
            stats[key] += 1
        if y == 1:
            key = 'tp' if yc == 1 else 'fn'
            stats[key] += 1

    precision = safe_division(stats['tp'], stats['tp'] + stats['fp'])
    recall = safe_division(stats['tp'], stats['tp'] + stats['fn'])
    f1 = safe_division(2 * precision * recall, precision + recall)

    print('precision  = %6.2f%%' % (100 * precision))
    print('recall     = %6.2f%%' % (100 * recall))
    print('F1 measure = %6.2f' % f1)

if __name__ == "__main__":
    main(0.1)
