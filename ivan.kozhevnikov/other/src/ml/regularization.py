import util
from svm import Svm
import numpy as np


def build_svm_get_result(input_data, c):
    np.random.shuffle(input_data)
    split_index = len(input_data) / 3
    learn_data = input_data[:split_index]
    check_data = input_data[split_index:]
    svm = Svm(c, *zip(*learn_data))
    return util.calc_score(check_data, svm)


def find_regularization_const(input_data):
    iterations = 3
    best_result = None
    best_c = None
    for power in range(-3, 5):
        c = 2 ** power
        f_metric = 0
        for iteration in range(iterations):
            f_metric += util.f1_metric(*build_svm_get_result(input_data, c))
        f_metric /= 3
        if best_result is None or f_metric > best_result:
            best_c = c
            best_result = f_metric
    return best_c



