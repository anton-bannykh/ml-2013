import math
from cancer_common.reader import *

LEARNING_RATE = 0.01
EPS = 0.1

def norm(x):
    return numpy.sqrt(numpy.inner(x, x))

def get_w_logistic_regression(x_in, y_in, const):
    w = numpy.zeros(len(x_in[0]))
    difPrev = 0
    while (True):
        wPrev = numpy.array(w)
        cur_x_in, cur_y_in = shuffle(x_in, y_in)
        for i in range(len(cur_x_in)):
            grad = numpy.zeros(len(x_in[0]))
            for j in range(len(x_in[0])):
                if cur_y_in[i] * numpy.inner(w, cur_x_in[i]) < 20:
                    grad[j] += cur_y_in[i] * cur_x_in[i][j] / (1 + math.exp(cur_y_in[i] * numpy.inner(w, cur_x_in[i])))
                if j > 0:
                   grad[j] += const * w[j]
            w += LEARNING_RATE * grad

        dif = w - wPrev
        if difPrev < norm(dif) and difPrev != 0:
            break
        else:
            difPrev = norm(dif)
        if (norm(dif) < EPS):
            break
    return w

def predict_logistic_regression(x_out, w):
    return [1.0 if 1 / (1 + math.exp(-numpy.inner(x, w))) >= 0.5 else -1.0 for x in x_out]

