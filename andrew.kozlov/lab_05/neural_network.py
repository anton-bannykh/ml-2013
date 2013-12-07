import numpy as np
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

    classified = classify(get_x(), theta1, theta2)
    for i in range(len(data)):
        (_, y) = data[i]

        if classified[i] != y:
            count += 1

    return count / len(data)


def classify(x, theta1, theta2):
    ones = np.ones(len(x))

    h1 = sigmoid(np.append(ones, x, axis=0) * theta1)
    h2 = sigmoid(np.append(ones, h1, axis=0) * theta2)

    return np.argmax(h2, axis=1)


def make1d(theta):
    return np.reshape(theta, theta.size)


def reshape(params, l_in, size):
    return params[:size * (l_in + 1)].reshape((size, l_in + 1)), params[size * (l_in + 1):].reshape((2, size + 1))


def sum_squared(theta):
    return np.sum(np.sum(theta ** 2))


def cost_function(x, y, lambda_c, params, l_in, size):
    theta1, theta2 = reshape(params, l_in, size)

    m = len(x)
    ones = np.array(np.ones((m, 1)))
    x = np.concatenate((ones, x), axis=1)

    j = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    a = sigmoid(x * theta1)
    ones = np.array(np.ones((len(a), 1)))
    a = np.concatenate((ones, a), axis=1)

    h = sigmoid(a * theta2)
    y_tmp = [(1, 0) if v == 0 else (0, 1) for v in y]

    for i in range(m):
        j -= y_tmp[i] * np.log(h[i]) - (1 - y_tmp[i]) * np.log(1 - h[i])
    j /= m

    theta1_new = [(0, v) for (_, v) in theta1]
    theta2_new = [(0, v) for (_, v) in theta2]

    j += lambda_c * (sum_squared(theta1_new) + sum_squared(theta2_new)) / (2 * m)

    for i in range(m):
        d3 = h[i] - y_tmp[i]

        tmp = theta2 * d3
        d2 = tmp[1:] * sigmoid_gradient(x[i] * theta1)

        theta1_grad += d2 * x[i]
        theta2_grad += d3 * a[i]

    return j, [make1d(theta1_grad), make1d(theta2_grad)]


def thetas(data, size, lambda_c):
    l_in = line_len(data)
    theta1 = initialize_weights(l_in, size)
    theta2 = initialize_weights(size, 2)

    cost = lambda p: cost_function(get_x(data), get_y(data), lambda_c, p, l_in, size)
    params = fmin(cost, np.append(theta1, theta2))

    return reshape(params, l_in, size)


def cross_validate(data, hidden_layer_size, lambda_c, n=5):
    step = len(data) // n

    result = 0
    for i in range(0, len(data), step):
        theta1, theta2 = thetas(data[i + step:] + data[:i], hidden_layer_size, lambda_c)
        result += calculate_error(data[i: i + step], theta1, theta2)

    return result / n


def optimize_size_lambda(data, size_max=20, base=2.0, degree_max=10, coefficient=100):
    result, error = (line_len(data), 0), 2.0

    for size in range(1, size_max):
        for degree in range(1, degree_max):
            lambda_c = (base ** degree) / coefficient

            average_error = cross_validate(data, size, lambda_c)

            if average_error < error:
                error = average_error
                result = (size, lambda_c)

    return result