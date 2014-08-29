import numpy as np
from urllib.request import urlopen
import svm_smo
from kernal import *

def load_data():
    lines = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data").readlines()
    n, d = len(lines), len(lines[0].decode('utf-8').split(',')) - 2
    x = np.zeros((n, d + 2))

    for i in range(n):
        line = lines[i].decode('utf-8').split(',')
        x[i] = np.array([float(f) for f in line[2:]] + [1.0, 1 if line[1] == 'M' else -1])

    return x

def split(data, fraction=0.20):
    np.random.shuffle(data)
    point = int(len(data) * fraction)
    return data[point:], data[:point]

def main():
    data = load_data()
    train, test = split(data)

    alpha, b = svm_smo.train(train, 200.0, kernel=gaussian)
    result = svm_smo.testing(train, test, (alpha, b), kernel=gaussian)
    
    print("Gaussian kernel")
    print("Total error: %.5f" % result['er'])
    print("Precision: %.5f" % result['pre'])
    print("Recall: %.5f" % result['rec'])
    print("F1-metric: %.5f" % result['f1'])

    alpha, b = svm_smo.train(train, 200.0, kernel=polynomial)
    result = svm_smo.testing(train, test, (alpha, b), kernel=polynomial)

    print("\nPolynomial kernel")
    print("Total error: %.5f" % result['er'])
    print("Precision: %.5f" % result['pre'])
    print("Recall: %.5f" % result['rec'])
    print("F1-metric: %.5f" % result['f1'])

if __name__ == '__main__':
    main()
