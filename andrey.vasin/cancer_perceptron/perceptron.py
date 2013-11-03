import numpy
import random


def classify(w, x):
    prod = numpy.inner(w, x)
    if prod > 0:
        return 1
    else:
        return -1


def linear_regression(x, y):
    return numpy.dot(numpy.linalg.pinv(x), y)


def misclassified(samples, w):
    result = []
    for (xCur, yCur) in samples:
        if (classify(w, xCur) != yCur):
            result.append((xCur, yCur))
    return result


def train_perceptron(x, y, iterations):
    w = linear_regression(x, y)
    samples = list(zip(x, y))

    cur_misclassified = misclassified(samples, w)
    best_w, best_result = list(w), len(cur_misclassified)

    for _ in range(iterations):
        if (best_result == 0):
            break

        xCur, yCur = random.choice(cur_misclassified)
        w += yCur * xCur

        cur_misclassified = misclassified(samples, w)
        if best_result > len(cur_misclassified):
            best_w = w.copy()
            best_result = len(cur_misclassified)

    return best_w, best_result / len(x)