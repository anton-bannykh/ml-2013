import numpy
from numpy.numarray import zeros
from lab_01.main import divide, percent, load_data
from lab_04.linear_regression import optimize_constant, linear_regression_w, calculate_error, line_length


__author__ = 'adkozlov'


def scale(data):
    m = line_length(data)
    x_min, x_max = numpy.array([1e10] * m), zeros(m)

    for (x, _) in data:
        for i in range(m):
            x_min[i] = min(x_min[i], x[i])
            x_max[i] = max(x_max[i], x[i])

    for (x, _) in data:
        for i in range(m):
            x[i] = (x[i] - x_min[i]) / (x_max[i] - x_min[i]) if x_max[i] != x_min[i] else 1

    return data


def main():
    train_set, test_set = divide(scale(load_data()), 0.2)

    c = optimize_constant(train_set)
    w = linear_regression_w(train_set, c)

    e = calculate_error(test_set, w)
    print('regularization constant = %f\nerror = %6.2f' % (c, percent(e)))


if __name__ == "__main__":
    main()
