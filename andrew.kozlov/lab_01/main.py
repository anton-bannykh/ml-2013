from urllib.request import urlopen
from lab_01.perceptron import *

__author__ = 'adkozlov'


def load_data():
    result = []

    file = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    for line in file.readlines():
        array = line.decode("utf-8").split(',')
        result.append(([1.0] + [float(f) for f in array[2:]], 1 if array[1] == "M" else -1))

    return result


def constants(data, fraction=0.1):
    test_len = int(len(data) * fraction)
    w = train(data[test_len:])

    results = {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0}
    for (x, y) in data[:test_len]:
        yc = classify(w, x)

        if y == 1:
            results['tp' if yc == 1 else 'fn'] += 1
        else:
            results['tn' if yc == -1 else 'fp'] += 1

    return results['tp'] / (results['tp'] + results['fp']), results['tp'] / (results['tp'] + results['fn'])  # (precision, recall)

if __name__ == "__main__":
    p, r = constants(load_data())
    print('precision = %6.2f\nrecall = %6.2f' % (100 * p, 100 * r))