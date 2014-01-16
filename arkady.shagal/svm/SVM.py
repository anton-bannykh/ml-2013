import numpy
from common.Reader import get_data
from cvxopt import matrix, solvers


def calc_error(x, y, w, b):
    prediction = predict(x, w, b)
    misclassified = 0
    for j in range(len(x)):
        if y[j] != prediction[j]:
            misclassified += 1
    return misclassified / len(x)

def transpose(ls):
    return list(map(list, zip(*ls)))

def getMatrices(x, y, C):
    P = numpy.identity(len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            P[i][j] = y[i] * y[j] * numpy.inner(x[i], x[j])
    A = [[float(y)] for (y) in y]
    b = [0.0]
    q = [-1.0] * len(x)
    G = numpy.zeros(shape=(len(x) * 2, len(x)))
    h = [0.0] * len(x) + [C] * len(x)
    for i in range(len(x)):
            G[i][i] = -1
    for i in range(len(x)):
            G[len(x) + i][i] = 1
    G = transpose(G)
    return [matrix(m) for m in [P, q, G, h, A, b]]

def get_lagrange_coef_reg(x, y, const):
    solvers.options['show_progress'] = False
    sol = solvers.qp(*getMatrices(x, y, const))['x']
    alpha = [sol[i] for i in range(len(x))]
    return alpha

def fit_svm(x, y, C, eps=1e-6):
    alpha = get_lagrange_coef_reg(x, y, C)

    w = numpy.zeros(len(x[0]))

    for i in range(len(x)):
        if alpha[i] > eps:
            w += alpha[i] * y[i] * numpy.array(x[i])

    b = 0
    for i in range(len(x)):
        if eps < alpha[i] < C - eps:
            b = y[i] - numpy.inner(w, x[i])
            break

    return w, b

def predict(x, w, b):
    return [1.0 if numpy.inner(w, x[i]) + b > 0 else -1.0 for i in range(len(x))]

def get_C(x, y, parts):
    part_size = int(len(x) / parts)
    minEout = 1
    result = 1

    for deg in range(-5, 5):
        const = (10.0) ** (deg)
        print(deg)
        Eout = []

        for i in range(parts):
            check_set_x = x[i * part_size: (i + 1) * part_size]
            check_set_y = y[i * part_size: (i + 1) * part_size]

            training_set_x = x[:i * part_size] + x[(i + 1) * part_size:]
            training_set_y = y[:i * part_size] + y[(i + 1) * part_size:]

            w, b = fit_svm(training_set_x, training_set_y, const)
            Eout.append(calc_error(check_set_x, check_set_y, w, b))

        average = numpy.average(Eout)
        if average < minEout:
            minEout = average
            result = const

    return result

TEST_PERCENT = 0.3
PARTS = 10

x, y = get_data()
test_size = int(len(x) * TEST_PERCENT)

check_set_x = x[:test_size]
check_set_y = y[:test_size]

training_set_x = x[test_size:]
training_set_y = y[test_size:]

const = get_C(training_set_x, training_set_y, PARTS)
w, b = fit_svm(training_set_x, training_set_y, const)
Ein = calc_error(training_set_x, training_set_y, w, b)

prediction = predict(check_set_x, w, b)

stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

for j in range(len(check_set_x)):
    if check_set_y[j] == -1:
        key = 'tn' if prediction[j] == -1 else 'fp'
        stats[key] += 1
    if check_set_y[j] == 1:
        key = 'tp' if prediction[j] == 1 else 'fn'
        stats[key] += 1

precision = stats['tp'] / (stats['tp'] + stats['fp'])
recall = stats['tp'] / (stats['tp'] + stats['fn'])
F1 = 2 * precision * recall / (precision + recall)
Eout = (stats['fp'] + stats['fn']) / len(x[:test_size])

print("C = %6.5f" % const)
print('in sample error         = %6.2f' % (100 * Ein))
print('out of sample error     = %6.2f' % (100 * Eout))
print('precision               = %6.2f' % (100 * precision))
print('recall                  = %6.2f' % (100 * recall))
print('F1                      = %6.2f' % (100 * F1))