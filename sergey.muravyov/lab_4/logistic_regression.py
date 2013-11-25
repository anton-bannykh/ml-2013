import math
import numpy
from urllib.request import urlopen

def get_data():
    x, y = [], []
    file = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    for line in file.readlines():
        input = line.decode('utf-8').strip().split(',')

        if input[1] == 'M':
            y.append(1.0)
        else:
            y.append(-1.0)

        parameters = [float(x) for x in input[2:]]
        parameters.insert(0, 1.0)
        parameters = numpy.array(parameters)
        x.append(parameters)
    file.close()
    
    x_min, x_max = numpy.array([1e10] * len(x[0])), numpy.zeros(len(x[0]))
    for x_cur in x:
        for i in range(len(x_cur)):
            if x_min[i] > x_cur[i]:
                x_min[i] = x_cur[i]
            if x_max[i] < x_cur[i]:
                x_max[i] = x_cur[i]
                
    for x_cur in x:
        for i in range(len(x_cur)):
            if (x_max[i] != x_min[i]):
                x_cur[i] = (x_cur[i] - x_min[i]) / (x_max[i] - x_min[i])
            else:
                x_cur[i] = 1
    return shuffle(x, y)

def shuffle(x, y):
    tmp = list(zip(x, y))
    numpy.random.shuffle(tmp)
    x_new = []
    y_new = []
    for i in range(len(tmp)):
        x_new.append(tmp[i][0])
        y_new.append(tmp[i][1])
    return x_new, y_new

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
            w += 0.01 * grad

        dif = w - wPrev
        if difPrev < norm(dif) and difPrev != 0:
            break
        else:
            difPrev = norm(dif)
        if (norm(dif) < 0.1):
            break
    return w

def predict_logistic_regression(x_out, w):
    return [1.0 if 1 / (1 + math.exp(-numpy.inner(x, w))) >= 0.5 else -1.0 for x in x_out]


def get_best_constant(x, y, parts):
    part_size = int(len(x) / parts)
    min_error = 1
    best_C = 1

    for deg in range(0, 10):
        C = 0.1 ** (deg)
        print(deg)
        error = 0

        for i in range(parts):
            check_set_x = x[i * part_size: (i + 1) * part_size]
            check_set_y = y[i * part_size: (i + 1) * part_size]

            training_set_x = x[:i * part_size] + x[(i + 1) * part_size:]
            training_set_y = y[:i * part_size] + y[(i + 1) * part_size:]

            w = get_w_logistic_regression(training_set_x, training_set_y, const)
            prediction = predict_logistic_regression(check_set_x, w)

            curr_error = 0
            for j in range(len(check_set_x)):
                if check_set_y[j] != prediction[j]:
                    curr_error += 1
            error += curr_error / len(check_set_x) / parts

        if error < min_error:
            min_error = error
            best_C = C

    return best_C

def main():
    
    x, y = get_data()
    test_size = int(len(x) * 0.2)
    
    check_set_x, check_set_y = x[:test_size], y[:test_size]
    xs, ys = x[test_size:], y[test_size:]

    C = get_best_constant(xs, ys, 10)

    w = get_w_logistic_regression(xs, ys, C)
    prediction = predict_logistic_regression(xs, w)

    mc = 0
    for j in range(len(xs)):
        if training_set_y[j] != prediction[j]:
            mc += 1
    Ein = mc / len(xs)
    prediction = predict_logistic_regression(check_set_x, w)

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

    print("regularization constant = %6.5f" % C)
    print('in sample error = %6.2f' % (100 * Ein))
    print('out of sample error = %6.2f' % (100 * Eout))
    print('precision = %6.2f' % (100 * precision))
    print('recall = %6.2f' % (100 * recall))
    print('f1 = %6.2f' % (100 * F1))
