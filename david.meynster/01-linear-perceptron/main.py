import numpy as np
from urllib.request import urlopen
import perceptron

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

def load_dataset():
    lines = urlopen(url).readlines()
    n, d = len(lines), len(lines[0].decode('utf-8').split(',')) - 2
    x = np.zeros((n, d + 2))

    for i in range(n):
        line = lines[i].decode('utf-8').split(',')
        x[i] = np.array([float(f) for f in line[2:]] + [1.0, 1 if line[1] == 'M' else -1])

    return x

def split(data, fraction = 0.2):
    np.random.shuffle(data)
    point = int(len(data) * fraction)
    return data[point:], data[:point]

def main():
    train, test = split(load_dataset())
    theta = perceptron.train(train[:, :-1], train[:, -1])
    stats = perceptron.test(test[:, :-1], test[:, -1], theta)

    precision = 100 * stats['tp'] / (stats['tp'] + stats['fp'])
    recall = 100 * stats['tp'] / (stats['tp'] + stats['fn'])
    error = 100 * (stats['fp'] + stats['fn']) / len(test)
    print("precision: %6.2f\nrecall: %6.2f\nerror: %6.2f\n" % (precision, recall, error))

if __name__ == '__main__':
    main()