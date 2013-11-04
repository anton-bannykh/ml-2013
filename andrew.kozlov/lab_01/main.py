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


def main(fraction=0.1):
    data = load_data()

    test_len = int(len(data) * fraction)
    w = train(data[test_len:])

    stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    for (x, y) in data[:test_len]:
        yc = classify(w, x)

        if y == 1:
            stats['tp' if yc == 1 else 'fn'] += 1
        else:
            stats['tn' if yc == -1 else 'fp'] += 1

    return stats['tp'] / (stats['tp'] + stats['fp']), stats['tp'] / (stats['tp'] + stats['fn'])  # (precision, recall)

if __name__ == "__main__":
    print('precision = %6.2f\nrecall = %6.2f' % main())