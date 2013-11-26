from copy import deepcopy
import numpy as np


class Perceptron:
    def __init__(self, data, iterations):
        # v[0] -- coorinates, v[1] -- label (class)
        self.theta = np.zeros(len(data[0][0]))
        for i in xrange(iterations):
            for v in data:
                if self.classify(v[0]) != v[1]:
                    self.theta += v[1] * v[0]

    def classify(self, vector):
        return np.sign(np.dot(self.theta.T, vector))

    def get_training_error(self, data):
        return np.count_nonzero([self.classify(v[0]) - v[1] for v in data]) / float(len(data))

    def get_prec_and_accur(self, data):
        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for v in data:
            if self.classify(v[0]) == v[1]:
                if v[1] == 1:
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if v[1] == 1:
                    false_neg += 1
                else:
                    false_pos += 1

        precision = float(true_pos) / (true_pos + false_pos)
        accuracy = float(true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        return precision, accuracy


def get_data(file_name):
    data_file = open(file_name, 'r')
    raw_data = data_file.readlines()
    data = np.array(
        [(np.array(list(map(float, s.split(',')[2:]))),
          1 if s.split(',')[1] == 'M' else -1)
         for s in raw_data])
    data_file.close()
    return data


def split_data(data, train_percent):
    shuffled = deepcopy(data)
    np.random.shuffle(shuffled)
    train_size = int(train_percent * len(data))
    return shuffled[:train_size], shuffled[train_size:]

data = get_data('wdbc.data')
train, test = split_data(data, 0.25)
classifier = Perceptron(train, 2000)

print("errors = " + str(classifier.get_training_error(test)))
pac = classifier.get_prec_and_accur(test)
print("precision = " + str(pac[0]))
print("accuracy = " + str(pac[1]))
