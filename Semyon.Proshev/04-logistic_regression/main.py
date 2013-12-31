import urllib
import random
import numpy
import math

def load_dataset():
    dataset_url = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    dataset_result = []    
    for line in dataset_url.readlines():
        record = line.split(',')
        record_result = 1 if record[1] == 'M' else -1
        record_evidence = [float(f) for f in record[2:]]
        dataset_result.append((record_evidence, record_result))
    length = len(dataset_result[0][0])
    xmin, xmax = numpy.array([1e10] * length), numpy.zeros(length)
    for (x, _) in dataset_result:
        for i in range(length):
            xmin[i] = min(xmin[i], x[i])
            xmax[i] = max(xmax[i], x[i])
    for (x, _) in dataset_result:
        for i in range(length):
            if xmax[i] == xmin[i]:
                x[i] = 1
            else:
                x[i] = (x[i] - xmin[i]) / (xmax[i] - xmin[i])
    return dataset_result

def split_set(dataset, percent):
    random.shuffle(dataset)
    boundary = int(percent * len(dataset))
    return dataset[:boundary], dataset[boundary:]

def cross_validate(data, c):
    step = len(data) // 10
    result = 0

    for i in range(0, len(data), step):
        w = linear_regression(data[i + step:] + data[:i], c)
        result += calculate_error(data[i:i + step], w)

    return result / 10

def linear_regression(data, c, exp_value=20, eps=0.1, rate=0.01):
    m = line_length(data)
    w = numpy.zeros(m)

    dif_prev = 0
    while True:
        w_prev = numpy.array(w)

        for (x, y) in data:
            grad = numpy.zeros(m)

            for j in range(m):
                v = y * numpy.inner(w, x)

                if v < exp_value:
                    grad[j] += function(v, y * x[j])
                if j != 0:
                    grad[j] += c * w[j]

            w += rate * grad

        dif_norm = norm(w - w_prev)
        if dif_prev < dif_norm and dif_prev != 0:
            break
        else:
            dif_prev = dif_norm
        if dif_norm < eps:
            break

    return w

def norm(w):
    return numpy.sqrt(numpy.inner(w, w))


def function(x, c=1):
    return c / (1 + math.exp(x))


def classify(x, w, margin=0.5):
    return 1.0 if function(-numpy.inner(x, w)) >= margin else -1.0


def calculate_error(data, w):
    count = 0

    for (x, y) in data:
        if classify(x, w) != y:
            count += 1

    return count / len(data)


def line_length(data):
    return len(data[0][0])

def get_best_c(data):
    result, error = 1, 1

    for d in range(20):
        c = 2 ** -d
        average_error = cross_validate(data, c)
        print("current constant = %f, current error = %f" % (c, average_error))

        if average_error < error:
            error = average_error
            result = c

    return result

def main():
    dataset = load_dataset()
    training_set, test_set = split_set(dataset, 0.2)
    
    c = get_best_c(training_set)
    w = linear_regression(training_set, c)

    e = calc_error(test_set, w)
    print("C: %.5f" % c)
    print("Errors: %.5f" % e)

if __name__ == "__main__":
	main()
