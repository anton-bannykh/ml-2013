import numpy as np
from urllib.request import urlopen
from cvxopt import matrix
from cvxopt.blas import dotu
from cvxopt.solvers import qp, options

def get_data():
    result = []

    file = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    for line in file.readlines():
        array = line.decode("utf-8").split(',')
        result.append(([1.0] + [float(f) for f in array[2:]], 1 if array[1] == "M" else -1))

    return result   

def kernel(xs, ys):
    return dotu(matrix(xs), matrix(ys))

def vector(n, i, v=1.0):
    result = [0 for _ in range(2 * n)]
    result[2 * i] = v
    result[2 * i + 1] = -v

    return result

def lagrange_coef(data, c):
    n = len(data)
    g, h = [], []
    p = [[y1 * y2 * kernel(x1, x2) for (x2, y2) in data] for (x1, y1) in data]
    q = [-1.0 for _ in data]
    for i in range(n):
        g.append(vector(n, i))
        
    for i in range(n):
        h.append(c)
        h.append(0.0)
    h = [h]
    a = [[float(y)] for (_, y) in data]
    b = [0.0]
    return [matrix(m) for m in [p, q, g, h, a, b]]


def get_w(data, c, eps=1e-6):
    x0, _ = data[0]
    options['show_progress'] = False
    solution = qp(*lagrange_coef(data, c))
    a = np.array(solution['x']).transpose()[0]
    w = np.zeros(len(x0))

    for i in range(len(data)):
        if a[i] < eps:
            continue

        x, y = data[i]
        w += y * a[i] * np.array(x)

    b = 0
    for i in range(len(data)):
        if 2 * eps < a[i] + eps < c:
            x, y = data[i]
            b = y - np.dot(w, x)
            break

    return w, b

def count_error(data, w, b):
    count = 0

    for (x, y) in data:
        cl = 1 if np.dot(w, x) + b > 0 else -1
        if cl != y:
            count += 1

    return count / len(data)


def cross_validate(data, c, n=10):
    step = len(data) // n
    errors = []

    for i in range(0, len(data), step):
        w, b = get_w(data[i + step:] + data[:i], c)
        errors.append(count_error(data[i:i + step], w, b))

    return np.average(errors)


def get_c(data, n=10):
    result, min_error = 0, 0

    for d in range(-n // 2, n // 2):
        c = 10 ** d
        error = cross_validate(data, c)

        if error < min_error or result == 0:
            min_error = error
            result = c

    return result

def main():
    data = get_data()
    np.random.shuffle(data)
    test_len = int(len(data) * 0.1)
    xs, ys = data[test_len:], data[:test_len]

    c = get_c(xs)
    w, b = fit_svm(xs, c)

    e = count_error(ys, w, b)
    print('C = %f' % c)
    print('error = %6.2f' % (100 * e))


if __name__ == "__main__":
    main()
