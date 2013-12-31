import scipy.optimize
from scipy.optimize import minimize
from tools import get_data_set, training_and_test_sets, error_rate

__author__ = 'jambo'
import numpy as np


def h(x, omega):
    return 1.0 / (1 + np.exp(-np.dot(x, omega)))


def logistic_cost_function(X, Y, regularization_parameter):
    m = len(Y)
    return lambda omega: (
        -1.0 / m * np.sum(Y * np.log(h(X, omega)) + (1 - Y) * (np.log(1 - h(X, omega))))
        + regularization_parameter / (2.0 * m) * np.dot(omega[1:], omega[1:])
    )


def get_answers_and_actual(X_test, Y_test, omega, ):
    actual = Y_test
    answers = [1 if h(x, omega) >= 0.5 else 0 for x in X_test]
    return answers, actual


def cross_validation(C, data_set, times=10):
    error_rates = []
    for _ in xrange(times):
        train_set, test_set = training_and_test_sets(data_set)
        X_tr = np.array([test.features[0] for test in train_set])
        Y_tr = np.array([1 if test.answer == 1 else 0 for test in train_set])
        X_test = np.array([test.features[0] for test in test_set])
        Y_test = np.array([1 if test.answer == 1 else 0 for test in test_set])
        cost_function = logistic_cost_function(X_tr, Y_tr, C)
        omega = np.zeros((1, len(X_tr[0]), ))
        result = minimize(cost_function, omega, jac=None)  # jac)
        omega = result.x
        rate = error_rate(*get_answers_and_actual(X_test, Y_test, omega))
        print rate
        error_rates.append(rate)
    return sum(error_rates) / float(len(error_rates))


def get_regularized_constant():
    regularization_list = []
    data_set = get_data_set(add_extra_feature=True)
    for c in (2 ** power for power in xrange(-10, 10)):
        print "C: %f" % c
        regularization_list.append((cross_validation(c, data_set), c))
    print sorted(regularization_list)
    return sorted(regularization_list)[0]


def main():
    """
    best error_rate = 0.013158, c = 0.062500
    """
    print "best error_rate = %f, c = %f" % get_regularized_constant()


if __name__ == '__main__':
    main()
