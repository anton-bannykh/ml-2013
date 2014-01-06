import math
import numpy
from common.Reader import shuffle, get_scaled_data


LEARNING_RATE = 0.01
EPS = 0.1


def predict(x_out, w):
    return [1.0 if (-numpy.inner(x, w)) <= 0 else -1.0 for x in x_out]

def calc_error(x, y, w):
    prediction = predict(x, w)
    misclassified = 0
    for j in range(len(x)):
        if y[j] != prediction[j]:
            misclassified += 1
    return misclassified / len(x)

def norm(x):
    return numpy.sqrt(numpy.inner(x, x))

def hypo(t, c=1):
    return c/float(1 + math.exp(t));

def get_w_logistic_regression(x_in, y_in, const):
    w = numpy.zeros(len(x_in[0]))
    difPrev = 0
    while (True):
        wPrev = numpy.array(w)
        x, y = shuffle(x_in, y_in)
        for i in range(len(x)):
            grad = numpy.zeros(len(x_in[0]))
            for j in range(len(x_in[0])):
                if y[i] * numpy.inner(w, x[i]) < 20:
                    grad[j] += y[i] * x[i][j] * hypo(y[i] * numpy.inner(w, x[i]))
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


def get_best_constant(x, y, parts):
    part_size = int(len(x) / parts)
    minEout = 1
    result = 1

    for deg in range(0, 10):
        const = (1.0 / 3.0) ** (deg)
        print(deg)
        Eout = []

        for i in range(parts):
            check_set_x = x[i * part_size: (i + 1) * part_size]
            check_set_y = y[i * part_size: (i + 1) * part_size]

            training_set_x = x[:i * part_size] + x[(i + 1) * part_size:]
            training_set_y = y[:i * part_size] + y[(i + 1) * part_size:]

            w = get_w_logistic_regression(training_set_x, training_set_y, const)
            Eout.append(calc_error(check_set_x, check_set_y, w))

        average = numpy.average(Eout)
        if average < minEout:
            minEout = average
            result = const

    return result

TEST_PERCENT = 0.3
PARTS = 10

x, y = get_scaled_data()
test_size = int(len(x) * TEST_PERCENT)

check_set_x = x[:test_size]
check_set_y = y[:test_size]

training_set_x = x[test_size:]
training_set_y = y[test_size:]

const = get_best_constant(training_set_x, training_set_y, PARTS)
w = get_w_logistic_regression(training_set_x, training_set_y, const)
Ein = calc_error(training_set_x, training_set_y, w)

prediction = predict(check_set_x, w)

result = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

for j in range(len(check_set_x)):
    if check_set_y[j] == -1:
        key = 'tn' if prediction[j] == -1 else 'fp'
        result[key] += 1
    if check_set_y[j] == 1:
        key = 'tp' if prediction[j] == 1 else 'fn'
        result[key] += 1

precision = result['tp'] / (result['tp'] + result['fp'])
recall = result['tp'] / (result['tp'] + result['fn'])
F1 = 2 * precision * recall / (precision + recall)
Eout = (result['fp'] + result['fn']) / len(x[:test_size])

print("regularization constant = %6.5f" % const)
print('in sample error         = %6.2f' % (100 * Ein))
print('out of sample error     = %6.2f' % (100 * Eout))
print('precision               = %6.2f' % (100 * precision))
print('recall                  = %6.2f' % (100 * recall))
print('F1                      = %6.2f' % (100 * F1))