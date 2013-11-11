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


def divide(data, fraction=0.1):
    np.random.shuffle(data)
    test_len = int(len(data) * fraction)
    return data[test_len:], data[:test_len]


def test(data, w):
    results = {'fp': 0, 'tp': 0, 'fn': 0, 'tn': 0}
    for (x, y) in data:
        yc = classify(w, x)

        if y == 1:
            results['tp' if yc == 1 else 'fn'] += 1
        else:
            results['tn' if yc == -1 else 'fp'] += 1

    p = results['tp'] / (results['tp'] + results['fp'])
    r = results['tp'] / (results['tp'] + results['fn'])
    e = (results['fp'] + results['fn']) / len(data)
    f_1 = 2 * p * r / (p + r)

    return p, r, e, f_1


def percent(f):
    return 100 * f


def main():
    train_set, test_set = divide(load_data())
    p, r, e, f_1 = test(test_set, train(train_set))

    print('precision = %6.2f\nrecall = %6.2f\nerror = %6.2f\nF_1 = %f' % (percent(p), percent(r), percent(e), f_1))


if __name__ == "__main__":
    main()