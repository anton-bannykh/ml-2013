import numpy as np
import random
from urllib.request import urlopen
import logistic_regration

def load_data():
    lines = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data").readlines()
    n, d = len(lines), len(lines[0].decode('utf-8').split(',')) - 2
    x = np.zeros((n, d + 2))

    for i in range(n):
        line = lines[i].decode('utf-8').split(',')
        x[i] = np.array([float(f) for f in line[2:]] + [1.0, 1 if line[1] == 'M' else -1])

    return x

def split(data, fraction=0.15):
    np.random.shuffle(data)
    point = int(len(data) * fraction)
    return data[point:], data[:point]

def main():
    data = load_data()
    train, test = split(data)
    best_lymbda = logistic_regration.optimize_regularization(data)
    print()
    print("Best Lymbda: %.5f" %  best_lymbda)

    theta = logistic_regration.train(train, lymbda= best_lymbda)
    result = logistic_regration.testing(test, theta)

    print("Total error: %.5f" % result['er'])
    print("Precision: %.5f" % result['pre'])
    print("Recall: %.5f" % result['rec'])
    print("F1-metric: %.5f" % result['f1'])

if __name__ == '__main__':
    main()
