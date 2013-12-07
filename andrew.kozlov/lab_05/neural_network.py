import numpy as np
from numpy.core.umath import log
from scipy.optimize import fmin
from lab_04.linear_regression import line_len

__author__ = 'adkozlov'


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def initialize_weights(l_in, size, eps=0.12):
    result = np.random.rand(size, l_in + 1)
    return 2 * result * eps - eps


def get_x(data):
    return np.array([x for (x, _) in data])


def get_y(data):
    return np.array([y for (_, y) in data])


def calculate_error(data, theta1, theta2):
    count = 0

    classified = classify(get_x(data), theta1, theta2)
    for i in range(len(data)):
        (_, y) = data[i]

        if classified[i] != y:
            count += 1

    return count / len(data)


def classify(x, theta1, theta2):
    ones = np.ones((len(x), 1))

    h1 = sigmoid(np.dot(np.concatenate((ones, x), axis=1), theta1))
    h2 = sigmoid(np.dot(np.concatenate((ones, h1), axis=1), theta2))

    return np.argmax(h2, axis=1)


def make1d(theta):
    return np.reshape(theta, theta.size)


def reshape(params, l_in, size, labels):
    return params[:size * (l_in + 1)].reshape((l_in + 1, size)), params[size * (l_in + 1):].reshape((size + 1, labels))


def sum_squared(theta):
    return np.sum(np.sum(theta ** 2))


def set_zero_column(theta):
    result = theta.copy()
    for l in result:
        l[0] = 0

    return result


def sum_squared(theta):
    return sum(sum(theta ** 2))


def calculate_j(h, lambda_c, m, theta1, theta2, y_tmp):
    j = 0
    for i in range(m):
        j -= np.dot(y_tmp[i], log(h[i])) - np.dot(1 - y_tmp[i], log(1 - h[i]))

    theta1_new = set_zero_column(theta1)
    theta2_new = set_zero_column(theta2)
    j += lambda_c * (sum_squared(theta1_new) + sum_squared(theta2_new)) / 2

    return j / m


def cost_function(x, y, lambda_c, params, l_in, size, labels):
    theta1, theta2 = reshape(params, l_in, size, labels)

    m = len(x)
    ones = np.ones((m, 1))
    x = np.concatenate((ones, x), axis=1)

    a = sigmoid(np.dot(x, theta1))
    ones = np.ones((len(a), 1))
    a = np.concatenate((ones, a), axis=1)

    h = sigmoid(np.dot(a, theta2))

    y_tmp = np.zeros((m, labels))
    for i in range(m):
        y_tmp[i][y[i]] = 1

    '''theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    for i in range(m):
        d3 = h[i] - y_tmp[i]
        tmp = np.dot(theta2, d3)
        d2 = tmp[1:] * sigmoid_gradient(np.dot(x[i], theta1))
        d2 = d2.reshape((1, len(d2)))

        theta1_grad += (d2 * x[i]).T
        theta2_grad += np.dot(d3, a[i])

    theta1_grad *= (theta1_grad + lambda_c * set_zero_column(theta1)) / m
    theta2_grad *= (theta2_grad + lambda_c * set_zero_column(theta2)) / m'''

    return calculate_j(h, lambda_c, m, theta1, theta2, y_tmp)


def thetas(data, size, lambda_c, labels):
    l_in = line_len(data)
    theta1 = initialize_weights(l_in, size)
    theta2 = initialize_weights(size, 2)

    cost = lambda p: cost_function(get_x(data), get_y(data), lambda_c, p, l_in, size, labels)
    init_p = np.concatenate((make1d(theta1), make1d(theta2)), axis=1)
    params = fmin(cost, init_p.reshape((len(init_p), 1)))

    return reshape(params, l_in, size, labels)


def cross_validate(data, hidden_layer_size, lambda_c, labels, n=5):
    step = len(data) // n

    result = 0
    for i in range(0, len(data), step):
        theta1, theta2 = thetas(data[i + step:] + data[:i], hidden_layer_size, lambda_c, labels)
        result += calculate_error(data[i: i + step], theta1, theta2)

    return result / n


def optimize_size_lambda(data, labels, size_max=20, base=2.0, degree_max=10, coefficient=100):
    result, error = (line_len(data), 0), 2.0

    for size in range(1, size_max):
        for degree in range(1, degree_max):
            lambda_c = (base ** degree) / coefficient

            average_error = cross_validate(data, size, lambda_c, labels)
            print('%d %f %f\n' % (size, lambda_c, average_error))

            if average_error < error:
                error = average_error
                result = (size, lambda_c)

    return result