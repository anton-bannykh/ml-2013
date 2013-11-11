import numpy as np
from random import choice
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

def get_data():
    x, y = [], []
    file = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    for line in file.readlines():
        fields = line.decode('utf-8').strip().split(',')
        y.append(1 if fields[1] == 'M' else 0)
        num = [float(i) for i in fields[2:]]
        num.insert(0, 1.0)
        x.append(num)
    file.close()
    return x, y

def linear_regression(xs, ys):
    return np.dot(np.linalg.pinv(xs), ys)

def misclassify(samples, w):
    result = []
    for (i, j) in samples:
        if (classify(w, i) != j):
            result.append((i, j))
    return result

def equal(x1, x2):
    return abs(x1 - x2) < 0.001

def classify(w, x):
    prod = np.inner(w, x)
    if prod > 0:
        return 1
    else:
        return -1

def train_perceptron(x, y, iteractions=1000):
    w = linear_regression(x, y)
    samples = list(zip(x, y))

    best_w, best_result = None, 0
    for _ in range(iteractions):
        mc = misclassify(samples, w)

        if best_w is None or best_result > len(mc):
            best_w = w.copy()
            best_result = len(mc)
        if best_result == 0:
            break

        x, y = choice(mc)
        w += y * np.array(x)

    return best_w

def main(test_fraction):
    x, y = get_data()
    test_size = int(len(x) * test_fraction)


    w = train_perceptron(x[:-test_size], y[:-test_size], 1000)

    stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for i, j in zip(x[-test_size:], y[-test_size:]):
        yc = classify(w, i)
        if j == -1:
            key = 'tn' if yc == -1 else 'fp'
            stats[key] += 1
        if j == 1:
            key = 'tp' if yc == 1 else 'fn'
            stats[key] += 1


    prec = (stats['tp']) / (stats['tp'] + stats['fp'])
    recall = (stats['tp']) / ( stats['tp'] + stats['fn'])
    F1 = 2 * prec * recall / (prec + recall)

    print('precision = %6.2f\nrecall = %6.2f\nF_1 = %f' % (100 * prec, 100 * recall, F1))

if __name__ == "__main__":
    main(0.1)
