import urllib
import random
import numpy
import math
import scipy.optimize

def load_dataset():
    dataset_url = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    dataset_result = []    
    for line in dataset_url.readlines():
        record = line.split(',')
        record_result = 1 if record[1] == 'M' else -1
        record_evidence = [float(f) for f in record[2:]]
        dataset_result.append((record_evidence, record_result))
    evidence_length = len(dataset_result[0][0])
    xmin, xmax = numpy.array([1e10] * evidence_length), numpy.zeros(evidence_length)
    for (x, _) in dataset_result:
        for i in range(evidence_length):
            xmin[i] = min(xmin[i], x[i])
            xmax[i] = max(xmax[i], x[i])
    for (x, _) in dataset_result:
        for i in range(evidence_length):
            if xmax[i] == xmin[i]:
                x[i] = 1
            else:
                x[i] = (x[i] - xmin[i]) / (xmax[i] - xmin[i])
    return dataset_result

def split_set(dataset, percent):
    random.shuffle(dataset)
    boundary = int(percent * len(dataset))
    return dataset[:boundary], dataset[boundary:]

def get_best_c(data):
    best_c, best_error = 1, 1
    delta = int(len(data) / 10)
    for d in range(15):
        c = 2 ** -d
        error = 0
        for i in range(0, len(data), delta):
            w = get_w(data[:i] + data[i + delta:], c)
            tp, tn, fp, fn = get_errors(data[i:i + delta], w)
            error += (float) (fp + fn) / delta
        error = error / 10
        print("error: %f" % error)
        print("c: %f" % c)
        if error < best_error:
            best_error = error
            best_c = c
    return best_c

def get_w(data, c):
    evidence_length = len(data[0][0])
    w = numpy.zeros(evidence_length)
    norm_prev = 0
    while True:
        w_prev = numpy.array(w)
        for (evidence, result) in data:
            grad = numpy.zeros(evidence_length)
            for i in range(evidence_length):
                v = result * numpy.inner(w, evidence)
                if v < 20:
                    grad[i] += g(-v, result * evidence[i])
            w += 0.01 * (grad + c * w)
        norm = numpy.sqrt(numpy.inner(w - w_prev, w - w_prev))
        if norm_prev < norm and norm_prev != 0:
            break
        else:
            norm_prev = norm
        if norm < 0.1:
            break
    return w

def g(x, c):
    return c / (1 + math.exp(-x))

def get_errors(data, w):
    tp, tn, fp, fn = 0, 0, 0, 0
    for (evidence, result) in data:
        is_pos = g(numpy.inner(evidence, w), 1) >= 0.5
        if is_pos and result == 1:
            tp += 1
        elif is_pos and result == -1:
            fp += 1
        elif not is_pos and result == 1:
            fn += 1
        else:
            tn += 1
    return tp, tn, fp, fn

def get_metrics(tp, tn, fp, fn):
    precision = tp / (float) (tp + fp)
    recall = recall = tp / (float) (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def main():
    dataset = load_dataset()
    training_set, test_set = split_set(dataset, 0.2)

    c = get_best_c(training_set)
    w = get_w(training_set, c)

    tp, tn, fp, fn = get_errors(test_set, w)
    print("RESULT:")
    print("TP", tp)
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)
    print("C", c)

    p, r, f1 = get_metrics(tp, tn, fp, fn)
    print("Precision: %.5f" % p)
    print("Recall: %.5f" % r)
    print("F1: %.5f" % f1)

if __name__ == "__main__":
	main()
