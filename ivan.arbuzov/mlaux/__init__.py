from urllib import urlopen
import numpy as np
from os.path import exists


class Statistic:
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0

    def __init__(self, pred=None, y=None):
        if pred is None or y is None:
            return
        for i in range(len(y)):
            if pred[i] == 1 and y[i] == 1:
                self.true_pos += 1
            elif pred[i] == 1 and y[i] == -1:
                self.false_pos += 1
            elif pred[i] == -1 and y[i] == 1:
                self.false_neg += 1
            else:
                self.true_neg += 1

    def precision(self):
        return float(self.true_pos) / (self.true_pos + self.false_pos)

    def recall(self):
        return float(self.true_pos) / (self.true_pos + self.false_neg)

    def error(self):
        return float(self.false_pos + self.false_neg) / (self.true_pos + self.false_pos + self.false_neg + self.true_neg)

    def f1(self, beta=1):
        p = self.precision()
        r = self.recall()
        return float(1 + beta) * p * r / (beta * beta * p + r)


def load_data(url="http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
              local_path="wdbc.data"):
    try:
        if not exists(local_path):
            local_copy = open(local_path, 'wb')
            local_copy.write(urlopen(url).read())
            local_copy.close()
        raw_data = open(local_path, "r")
    except:
        raw_data = urlopen(url)
    vectors = raw_data.readlines()
    n, dim = len(vectors), len(vectors[0].split(',')) - 2
    X, Y = np.empty((n, dim)), np.empty(n)
    for i in range(n):
        vector = vectors[i].split(',')
        X[i] = np.array([float(f) for f in vector[2:]])
        Y[i] = 1 if vector[1] == 'M' else -1
    return X, Y


def split(data, frac=0.2, sh=True):
    if sh:
        np.random.shuffle(data)
    m = int(frac * len(data))
    return data[m:], data[:m]


def split_xy(data):
    return data[:, :-1], data[:, -1]


def grouped(dataset):
    x, y = dataset
    return np.concatenate((x, y.reshape((len(y), 1))), axis=1)
