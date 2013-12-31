from urllib.request import urlopen
import numpy as np

class Vector:
    def __init__(self, data, label):
        self.data = data
        self.label = label


def split_data(data, fraction):
    perm = np.random.permutation(data.size)
    data1_size = int(fraction * data.size)
    data1, data2 = data[perm[:data1_size]], data[perm[data1_size:]]

    return data1, data2


def load_data(train_fraction):
    try:
        inf = open('wdbc.data')
    except IOError:
        DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        inf = urlopen(DATA_URL)

    ss = inf.readlines()
    vecs = np.array([Vector(np.array(list(map(float, s.split(',')[2:]))), 1 if s.split(',')[1] == 'M' else -1) for s in ss])
    tr, t = split_data(vecs, train_fraction)
    inf.close()
    return (tr, t)

def get_training_error(classifier, tests):
    return np.count_nonzero([abs(classifier.get_label(v.data) - v.label) for v in tests]) / len(tests)

def division_with_zero(a, b):
    return 0 if b == 0 else a / b

def get_metrics(classifier, tests):
        tp = fp = fn = tn = 0
        for v in tests:
            if classifier.get_label(v) == v.label:
                if v.label == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if v.label == 1:
                    fn += 1
                else:
                    fp += 1
        precision = division_with_zero(tp, (tp + fp))
        accuracy = division_with_zero((tp + tn), (tp + tn + fp + fn))
        recall = division_with_zero(tp, (tp + fn))
        f1 = division_with_zero(2 * precision * recall,  (precision + recall))

        return {'precision' : precision, 'accuracy' : accuracy, 'recall' : recall, 'f1' : f1}

def print_results(classifier, test):
    print("Training error = " + str(get_training_error(classifier, test)))
    m = get_metrics(classifier, test)
    print("Precision = " + str(m['precision']))
    print("Accuracy = " + str(m['accuracy']))
    print("Recall = " + str(m['recall']))
    print("F1 = " + str(m['f1']))