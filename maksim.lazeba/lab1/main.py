__author__ = 'max'

from perceptron import *


def get_input(data_file, train_input_percent):
    file = open(data_file, "rU")
    instances = []
    for line in file:
        split = line.split(",")
        instance_id = int(split[0])
        if split[1] == 'M':
            diagnosis = 1
        else:
            diagnosis = -1
        features = [float(f) for f in split[2:]]
        instance = (instance_id, diagnosis, features)
        instances.append(instance)
    file.close()

    import random as r

    r.shuffle(instances)
    train_input_num = int((train_input_percent / 100) * len(instances))
    return instances[:train_input_num], instances[train_input_num + 1:]


def main():
    training, test = get_input("data/wdbc.data", 80)
    dim = len(training[0][2])
    p = get_untraining_perceptron(dim)
    p.train(0.1, training, 30)

    print(p.weights)

    tp = 0
    fp = 0
    fn = 0
    for x in test:
        t = x[1]
        out = p.calc_output(x[2])
        if t > 0 and out > 0:
            tp += 1
        elif t > 0 > out:
            fn += 1
        elif t < 0 < out:
            fp += 1
    print("Perceptron results")
    print("Precision: ", tp / (tp + fp))
    print("Recall: ", tp / (tp + fn))


if __name__ == "__main__":
    main()
