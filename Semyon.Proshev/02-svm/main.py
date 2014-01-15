import urllib
import random
import numpy
import scipy.optimize

def load_dataset():
    dataset_url = urllib.urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    dataset_result = []    
    for line in dataset_url.readlines():
        record = line.split(',')
        record_result = 1 if record[1] == 'M' else -1
        record_evidence = [float(f) for f in record[2:]]
        dataset_result.append((record_evidence, record_result))
    return dataset_result

def split_set(dataset, percent):
    random.shuffle(dataset)
    boundary = int(percent * len(dataset))
    return dataset[:boundary], dataset[boundary:]

def svm_training(dataset, c):
    dimension = len(dataset[0][0])
    x, y = zip(*dataset)
    def svm_f(w):
        w0, w1 = w[-1], w[:-1]
        classified = y * (numpy.dot(x, w1) + w0)
        return 0.5 * sum(w1 * w1) + c * sum(1 - classified[classified < 1])
    return scipy.optimize.fmin(svm_f, numpy.zeros(dimension + 1))

def svm_classify(dataset, w):
    w0, w1 = w[-1], w[:-1]
    classified = numpy.dot(zip(*dataset)[0], w1) + w0
    classified[classified < 0], classified[classified >= 0] = -1, 1
    return classified
    

def svm_testing(dataset, w):
    tp, tn, fp, fn = 0, 0, 0, 0
    classified = svm_classify(dataset, w)
    for i in range(len(dataset)):
        if classified[i] == 1 and dataset[i][1] == 1:
            tp += 1
        elif classified[i] == 1 and dataset[i][1] == -1:
            fp += 1
        elif classified[i] == -1 and dataset[i][1] == 1:
            fn += 1
        else:
            tn += 1
    return tp, tn, fp, fn

def count_svm(dataset, c):
    training_set, test_set = split_set(dataset, 0.2)
    w = svm_training(training_set, c)
    tp, tn, fp, fn = svm_testing(test_set, w)
    return tp, tn, fp, fn

def count_metrics(tp, tn, fp, fn):
    precision = tp / (float) (tp + fp)
    recall = recall = tp / (float) (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def main():
    dataset = load_dataset()
    opt_f1 = 0
    opt_c = 0
    for degree in range(0, 20):
        c = 0.2 ** degree
        tp, tn, fp, fn = count_svm(dataset, c)
        print("TP", tp)
        print("TN", tn)
        print("FP", fp)
        print("FN", fn)
        print("C", c)
        if (tp + fp == 0 or tp + fn == 0):
            continue
        p, r, f1 = count_metrics(tp, tn, fp, fn)
        if f1 > opt_f1:
            opt_f1 = f1
            opt_c = c

    tp, tn, fp, fn = count_svm(dataset, opt_c)
    print("TP", tp)
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)
    print("C", opt_c)
    p, r, f1 = count_metrics(tp, tn, fp, fn)
    print("Precision: %.5f" % p)
    print("Recall: %.5f" % r)
    print("F1: %.5f" % f1)

if __name__ == "__main__":
	main()
