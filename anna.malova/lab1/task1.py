import numpy as np

class Vector:
    def __init__(self, data, label):
        self.data = data
        self.label = label

class Perceptron:
    def __init__(self, vec, steps):
        self.theta = np.zeros(vec[0].data.size)
        for i in range(steps):
            for v in vec:
                if self.get_label(v) != v.label:
                    self.theta += v.label * v.data
        print("Training completed")


    def get_label(self, v):
        return np.sign(np.dot(self.theta.T, v.data))

    def get_training_error(self, vec):
        return np.count_nonzero([self.get_label(v.data) - v.label] for v in vec) / len(vec)

    def get_accuracy(self, vec):
        tp = fp = fn = tn = 0
        for v in vec:
            if self.get_label(v) == v.label:
                if v.label == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if v.label == 1:
                    fn += 1
                else:
                    fp += 1

        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return (precision, accuracy)


def read_data(train_percent):
    inf = open('wdbc.data')
    ss = inf.readlines()
    vecs = np.array([Vector(np.array(list(map(float, s.split(',') [2:]))), 1 if s.split(',')[1] == 'M' else -1) for s in ss])
    perm = np.random.permutation(vecs.size)
    train_size = int(train_percent * vecs.size)
    tr, t = perm[:train_size], perm[train_size:]
    inf.close()

    return (vecs[tr], vecs[t])

train, test = read_data(0.3)
classifier = Perceptron(train, 1000)

print("Training error = " + str(classifier.get_training_error(test)))
ac = classifier.get_accuracy(test)
print("Precision = " + str(ac[0]))
print("Accuracy = " + str(ac[1]))
