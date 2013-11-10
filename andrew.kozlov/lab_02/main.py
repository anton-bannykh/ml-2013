from numpy import average
from sklearn import svm

from lab_01.main import divide, percent, load_data


__author__ = 'adkozlov'


def unzip(lst):
    return zip(*lst)


def optimize_c(data, n=10):
    step = len(data) // n

    result, error = 0, 0
    for d in range(15):
        c = 10 ** (-d)

        errors = []
        for i in range(0, len(data), step):
            errors.append(calculate_error(data[i:i + step], fit_svm(data[i + step:] + data[:i], c)))

        average_error = average(errors)
        if error > average_error or result == 0:
            error = average_error
            result = c

    return result


def fit_svm(data, c):
    result = svm.LinearSVC(C=c)
    xs, ys = unzip(data)
    result.fit(xs, ys)

    return result


def calculate_error(data, svc):
    result = 0
    for (x, y) in data:
        if y != svc.predict(x):
            result += 1

    return result / len(data)


def main():
    train_set, test_set = divide(load_data())
    c = optimize_c(train_set)
    e = calculate_error(test_set, fit_svm(train_set, c))
    print('C = %f\nerror = %6.2f' % (c, percent(e)))


if __name__ == "__main__":
    main()