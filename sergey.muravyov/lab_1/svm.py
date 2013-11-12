from math import ceil
from math import log
from sklearn import svm
import numpy as np
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

def shuffle(x, y):
    tmp = list(zip(x, y))
    np.random.shuffle(tmp)
    x_new, y_new = [], []
    for i in range(len(tmp)):
        x_new.append(tmp[i][0])
        y_new.append(tmp[i][1])
    return x_new, y_new

def get_data():
    x, y = [], []
    file = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    for line in file.readlines():
        fields = line.decode('utf-8').strip().split(',')
        y.append(1 if fields[1] == 'M' else -1)
        num = [float(i) for i in fields[2:]]
        num.insert(0, 1.0)
        x.append(num)
    file.close()
    return shuffle(x, y)


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

def best_regularization(x, y, parts=5):
    part_size = int(len(x) / parts)
    best_error = 1
    best_C = 0

    for d in range(200):
        const = (1.0 / 9.0) ** (d)
        curr_error = 0

        for i in range(parts):
            check_set_x = x[i * part_size: (i + 1) * part_size]
            check_set_y = y[i * part_size: (i + 1) * part_size]

            training_set_x = x[:i * part_size] + x[(i + 1) * part_size:]
            training_set_y = y[:i * part_size] + y[(i + 1) * part_size:]

            model = svm.LinearSVC(C=const)
            model.fit(training_set_x, training_set_y)

            prediction = model.predict(check_set_x)

            misclassified = 0
            for j in range(len(check_set_x)):
                if check_set_y[j] != prediction[j]:
                    misclassified += 1
            curr_error += misclassified / len(check_set_x) / parts

        if curr_error < best_error or best_C == 0:
            best_error = curr_error
            best_C = const

    return best_C

def main():
    xs, ys = get_data()
    test_size = int(len(xs) * 0.1)
    train_xs = xs[:test_size]
    train_ys = ys[:test_size]

    C = best_regularization(train_xs, train_ys, 10)

    model = svm.LinearSVC(C=C)
    model.fit(train_xs, train_ys)

    errors = 0
    for x, x_pred in zip(list(ys[-test_size:]), list(model.predict(xs[-test_size:]))):
        if x != x_pred:
            errors += 1
    print("error on test set: %6.5f%%" % (100 * errors / test_size))
    print("regularization constant is %6.2f" % C)

if __name__ == "__main__":
    main()