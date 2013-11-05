from math import ceil
from math import log
from sklearn import svm
import numpy as np
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

##get from dataset
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

def split(ls, parts):
    n = len(ls)
    part_size = int(ceil(n / parts))
    for i in range(parts):
        yield ls[i * part_size:(i + 1) * part_size]

def append(args):
    result = []
    for ls in args:
        result.extend(ls)
    return result

def average(ls):
    if len(ls) == 0:
        return 0
    return sum(ls) / len(ls)

def unzip(*args):
    return list(zip(*args))

def best_regularization(xs, ys, parts=5):
    splitted_data = list(split(unzip(xs, ys), parts))
    best_C, best_error = None, None
    for deg in range(10):
        C = 0.1**deg
        errors = []
        for i in range(parts):
            test_data = splitted_data[i]
            train_data = append(splitted_data[:i] + splitted_data[i+1:])

            xs, ys = zip(*train_data)
            model = svm.LinearSVC(C=C)
            model.fit(xs, ys)

            curr_error = 0.0
            xs, ys = zip(*test_data)
            predictions = model.predict(xs)
            for y, y_pred in zip(ys, predictions):
                if y != y_pred:
                    curr_error += 1
            errors.append(curr_error / len(test_data))

        curr_error = average(errors)
        if best_C is None or best_error > curr_error:
            best_error = curr_error
            best_C = C

    return best_C

def main():
    xs, ys = get_data()
    test_size = int(len(xs) * 0.1)
    train_xs = xs[:-test_size]
    train_ys = ys[:-test_size]
    C = best_regularization(train_xs, train_ys)
    model = svm.LinearSVC(C=C)
    model.fit(train_xs, train_ys)

    errors = 0
    for x, x_pred in zip(list(ys[-test_size:]), list(model.predict(xs[-test_size:]))):
        if x != x_pred:
            errors += 1
    print("error on test set: %6.2f%%" % (100 * errors / test_size))
    print("regularization constant is 1e-%d" % round(log(C, 0.1)))

if __name__ == "__main__":
    main()