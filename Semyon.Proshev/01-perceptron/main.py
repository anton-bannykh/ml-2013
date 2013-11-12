import urllib
import random
import numpy

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

def perceptron_classify(w, x):
    return 1 if numpy.dot(w, x) >= 0 else -1

def perceptron_training(dataset):
    n = len(dataset)
    dimension = len(dataset[0][0])
    w = numpy.zeros(dimension)
    for i in range(100):
        for j in range(n):
            if perceptron_classify(w, dataset[j][0]) != dataset[j][1]:
                for k in range(dimension):
                    w[k] += dataset[j][1] * dataset[j][0][k]
    return w
    
def perceptron_testing(dataset, w):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(dataset)):
        classified = perceptron_classify(w, dataset[i][0])
        if classified == 1 and dataset[i][1] == 1:
            tp += 1
        elif classified == 1 and dataset[i][1] == -1:
            fp += 1
        elif classified == -1 and dataset[i][1] == 1:
            fn += 1
        else:
            tn += 1
    return tp, tn, fp, fn

def main():
    dataset = load_dataset()
    training_set, test_set = split_set(dataset, 0.2)
    w = perceptron_training(training_set)
    tp, tn, fp, fn = perceptron_testing(test_set, w)

    precision = tp / (float) (tp + fp)
    recall = tp / (float) (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("TP", tp)
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)
    print("Precision: %.5f" % precision)
    print("Recall: %.5f" % recall)
    print("F1: %.5f" % f1)

if __name__ == "__main__":
	main()
