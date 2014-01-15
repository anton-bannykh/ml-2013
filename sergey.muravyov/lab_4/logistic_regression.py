import numpy
from math import exp
from numpy.numarray import zeros
from urllib.request import urlopen

def llength(data):
    x, _ = data[0]
    return len(x)

def get_data():
    result = []

    file = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    for line in file.readlines():
        array = line.decode("utf-8").split(',')
        result.append(([1.0] + [float(f) for f in array[2:]], 1 if array[1] == "M" else -1))

    rlen = llength(result)
    x_min, x_max = numpy.array([1e10] * rlen), zeros(rlen)
    for (x, _) in result:
        for i in range(rlen):
            x_min[i] = min(x_min[i], x[i])
            x_max[i] = max(x_max[i], x[i])
    for (x, _) in result:
        for i in range(rlen):
            x[i] = (x[i] - x_min[i]) / (x_max[i] - x_min[i]) if x_max[i] != x_min[i] else 1

    return result


def norm(x):
    return numpy.sqrt(numpy.inner(x, x))

def function(x, c=1):
    return c / (1 + exp(x))

def calculate_error(data, w):
    count = 0
    for (x, y) in data:
        cl = 1.0 if function(-numpy.inner(x, w)) >= 0.5 else -1.0
        if cl != y:
            count += 1
    return count / len(data)

def linear_regression_w(data, c, exp_value=20, eps=0.1, rate=0.01):
    m = llength(data)
    w = numpy.zeros(m)
    dif_prev = 0
    while True:
        w_prev = numpy.array(w)
        for (x, y) in data:
            grad = numpy.zeros(m)
            for j in range(m):
                v = y * numpy.inner(w, x)
                if v < exp_value:
                    grad[j] += function(v, y * x[j])
                if j != 0:
                    grad[j] += c * w[j]
            w += rate * grad
        dif_norm = norm(w - w_prev)
        if dif_prev < dif_norm and dif_prev != 0:
            break
        else:
            dif_prev = dif_norm
        if dif_norm < eps:
            break

    return w

def cross_validate(data, c, n=10):
    step = len(data) // n
    result = 0
    for i in range(0, len(data), step):
        w = linear_regression_w(data[i + step:] + data[:i], c)
        result += calculate_error(data[i:i + step], w)
    return result / n


def get_constant(data, base=3, n=10):
    result, min_error = 1, 1

    for d in range(n):
        c = base ** -d
        error = cross_validate(data, c)
        print("current deg = %d, current const = %f, current err = %f" % (d, c, error))
        if error < min_error:
            min_error = error
            result = c

    return result

def main():
    data = get_data()
    numpy.random.shuffle(data)
    test_len = int(len(data) * 0.2)
    xs, ys = data[test_len:], data[:test_len]

    c = get_constant(xs)
    e = calculate_error(ys, w)
    print('regularization constant = %f' % c)
    print('error = %6.2f' % (100 * e))

if __name__ == "__main__":
    main()
