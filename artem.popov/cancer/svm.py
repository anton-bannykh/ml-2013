import random
import datetime
import numpy
from tools import training_and_test_sets, error_rate, get_data_set

__author__ = 'jambo'


def get_answers_and_actual(X_tr, Y_tr, X_test, Y_test, alpha, beta, kernel_func=lambda x, y: x.dot(y.T)):
    actual = Y_test
    answers = [1 if svm_func(alpha, beta, X_tr, Y_tr, x, kernel_func) >= 0 else -1 for x in X_test]
    return answers, actual


def cross_validation(c, data_set, kernel_func=lambda x, y: x.dot(y.T), times=10):
    error_rates = []
    for _ in xrange(times):
        train_set, test_set = training_and_test_sets(data_set)
        X_tr = numpy.array([test.features[0] for test in train_set])
        Y_tr = numpy.array([test.answer for test in train_set])
        X_test = numpy.array([test.features[0] for test in test_set])
        Y_test = numpy.array([test.answer for test in test_set])
        alpha, beta = smo2(c, X_tr, Y_tr, kernel_func=kernel_func)
        rate = error_rate(*get_answers_and_actual(X_tr, Y_tr, X_test, Y_test, alpha, beta, kernel_func))
        print rate
        error_rates.append(rate)
    return sum(error_rates) / float(len(error_rates))


def get_regularized_constant(kernel_func=lambda x, y: x.dot(y.T)):
    regularization_list = []
    data_set = get_data_set()
    for c in (10 ** power for power in xrange(0, 1)):
        print c
        regularization_list.append((cross_validation(c, data_set, kernel_func), c))
    print sorted(regularization_list)
    return sorted(regularization_list)[0]


def svm_func(alpha, beta, X, Y, x, kernel_func=lambda X, y: X.dot(y.T)):
    return beta + numpy.sum(alpha * Y * kernel_func(X, x))


def smo2(C, X, Y, tolerance=10e-2, max_passes=10, kernel_func=lambda x, y: x.dot(y.T)):
    """

    @param C: regularization parameter
    @type training_set: list of DataElement
    @return:
    @rtype:

    http://cs229.stanford.edu/materials/smo.pdf
    """
    m = len(Y)
    alpha = numpy.zeros(m)
    beta = 0.0
    passes = 0
    iterate_all = 1
    while passes < max_passes:
        num_changed_alphas = 0
        for i in xrange(m):
            if not iterate_all and (abs(alpha[i]) < 10e-5 or abs(alpha[i] - C) < 10e-5):
                continue
            x_i = X[i]
            y_i = Y[i]
            error_i = svm_func(alpha, beta, X, Y, x_i, kernel_func) - y_i
            if not (((y_i * error_i < -tolerance and alpha[i] < C) or
                         (y_i * error_i > tolerance and alpha[i] > 0))):
                continue
            j = i
            while j == i:
                j = random.randint(0, m - 1)
            x_j = X[j]
            y_j = Y[j]
            error_j = svm_func(alpha, beta, X, Y, x_j, kernel_func) - y_j

            if (error_i + tolerance < error_j - tolerance and
                    (alpha[i] < C and y_i == 1 or alpha[i] > 0 and y_i == -1)
                and (alpha[j] < C and y_j == -1 or alpha[j] > 0 and y_j == 1)):

                eta = (error_j - error_i) / (-2 * kernel_func(x_i, x_j) + kernel_func(x_i, x_i) + kernel_func(x_j, x_j))

                if alpha[i] + y_i * eta > C:
                    eta = (C - alpha[i]) / y_i
                if alpha[j] - y_j * eta > C:
                    eta = (C - alpha[j]) / -y_j
                if alpha[i] + y_i * eta < 0:
                    eta = -alpha[i] / y_i
                if alpha[j] - y_j * eta < 0:
                    eta = alpha[j] / y_j

                alpha[i] += y_i * eta
                alpha[j] -= y_j * eta

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
        iterate_all = (1 - iterate_all)

    b1 = max([
        svm_func(alpha, beta, X, Y, X[i], kernel_func) for i
        in filter(lambda i: Y[i] == 1 and alpha[i] > 0, range(0, m))
    ])

    b2 = min([
        svm_func(alpha, beta, X, Y, X[i], kernel_func) for i
        in filter(lambda i: Y[i] == -1 and alpha[i] > 0, range(0, m))
    ])

    beta = (b1 + b2) / 2.0

    return alpha, beta


def polynominal_kernel(d):
    return lambda X, x: X.dot(x.T) ** d


def gaussian_kernel(gamma):
    return lambda X, x: numpy.exp(-gamma * numpy.linalg.norm(X - x) ** 2)


def analyze_C_and_kernels():
    for d in xrange(1, 4):
        best_result, constant = get_regularized_constant(polynominal_kernel(d))
        print "Regularized constant for polynomal kernel with d = %d: %d" % (d, constant)
        print "Best mean error_rate: %0.5f" % best_result
    for gamma in [10 ** x for x in xrange(-5, 5)]:
        best_result, constant = get_regularized_constant(gaussian_kernel(gamma))
        print "Regularized constant for gaussian kernel with gamma = %0.5f: %d" % (gamma, constant)
        print "Best mean error_rate: %0.5f" % best_result


def main():
    """
    Best result gives polynominal kernel with d = 1 and C = 100
    mean error_rate: 0.00526

    Also, polynomal kernel with d = 3 and C = 1000 gives decent result
    mean error_rate: 0.06316

    Gaussian kernels works pretty bad, i haven't found any (gamma, C) with decent error rate.

    to repeat C and kernels analysis, uncomment analyze_C_and_kernels()
    """
    # analyze_C_and_kernels()

    # time = datetime.datetime.now()
    C = 100
    print "error_rate mean: %0.5f" % cross_validation(C, get_data_set(), polynominal_kernel(1))
    # print (datetime.datetime.now() - time).microseconds


if __name__ == '__main__':
    main()


