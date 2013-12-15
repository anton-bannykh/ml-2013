import util
import numpy as np


def build_classifier_get_result(input_data, c, builder):
    np.random.shuffle(input_data)
    split_index = int(len(input_data) * 0.2)
    learn_data = input_data[:split_index]
    check_data = input_data[split_index:]
    vectors, classes = zip(*learn_data)
    smo = builder(c, vectors, classes)
    return util.calc_score(check_data, smo)


def find_regularization_const(input_data, builder):
    iterations = 3
    best_result = None
    best_c = None
    for power in range(-3, 4):
        c = 2 ** power
        print("Trying C: " + str(c))
        f_metric = 0
        for iteration in range(iterations):
            f_metric += util.f1_metric(*build_classifier_get_result(input_data, c, builder))
        f_metric /= 3
        if best_result is None or f_metric > best_result:
            best_c = c
            best_result = f_metric
    return best_c



