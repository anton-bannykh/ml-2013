import numpy as np
import random
from urllib.request import urlopen
import perceptron

random.seed("cancer")

def load_data():
    lines = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data").readlines()
    n, d = len(lines), len(lines[0].decode('utf-8').split(',')) - 2
    x = np.zeros((n, d + 2))

    for i in range(n):
        line = lines[i].decode('utf-8').split(',')
        x[i] = np.array([float(f) for f in line[2:]] + [1.0, 1 if line[1] == 'M' else -1])

    return x

def split(data, fraction = 0.15):
    random.shuffle(data)
    point = int(len(data) * fraction)
    return data[point:], data[:point]

def main():
    train, test = split(load_data())
    theta = perceptron.training(train[:, :-1], train[:, -1])
    stats = perceptron.testing(test[:, :-1], test[:, -1], theta)

    precision = stats['tp'] / (stats['tp'] + stats['fp'])
    recall = stats['tp'] / (stats['tp'] + stats['fn'])
    error = (stats['fp'] + stats['fn']) / len(test)
    f1 = 2 * precision * recall / (precision + recall)
    print("Total error: %.5f" % error)
    print("Precision: %.5f" % precision)
    print("Recall: %.5f" % recall)
    print("F1-metric: %.5f" % f1)

if __name__ == '__main__':
    main()