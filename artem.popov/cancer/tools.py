import random
import urllib2
import csv
import numpy

__author__ = 'jambo'

class DataElement:
    def __init__(self, data_list, add_extra_feature=False):
        self.id = int(data_list[0])
        self.answer = 1 if data_list[1] == 'M' else -1
        self.features = numpy.array([[1] + map(float, data_list[2:])])


def load_data_from_file():
    with open("wdbc.data", "r") as data_file:
        return list(csv.reader(data_file))
    pass


def download_data_file():
    data_file = urllib2.urlopen(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    output = file("wdbc.data", "w")
    output.write(data_file.read())
    output.close()


def load_cancer_dataset():
    try:
        return load_data_from_file()
    except:
        download_data_file()
        return load_data_from_file()


def get_data_set(add_extra_feature=False):
    dataset = [DataElement(list, add_extra_feature=add_extra_feature) for list in load_cancer_dataset()]
    l = len(dataset[0].features[0])
    for i in xrange(l):
        if add_extra_feature and i == 0:
            continue
        feature_values = [test.features[0][i] for test in dataset]
        mean = sum(feature_values) / float(len(feature_values))
        norm = max(feature_values) - min(feature_values)
        for test in dataset:
            test.features[0][i] = (test.features[0][i] - mean) / norm
    return dataset


def training_and_test_sets(dataset):
    """
    @rtype: list of DataElement, list of DataElement
    """
    dataset_ = list(dataset)
    random.shuffle(dataset_)
    separate_point = int(len(dataset_) * 0.8)
    return dataset_[:separate_point], dataset[separate_point:]


def error_rate(answers, actual):
    pairs = zip(answers, actual)
    mistakes = len([1 for pair in pairs if pair[0] != pair[1]])
    return float(mistakes) / len(pairs)


def precision(answers, actual):
    pairs = zip(answers, actual)
    answered = len([1 for pair in pairs if pair[0] == 1])
    good = len([1 for pair in pairs if pair[0] == 1 and pair[1] == 1])
    if not answered: return 0
    return float(good) / answered


def recall(answers, actual):
    pairs = zip(answers, actual)
    expected = len([1 for pair in pairs if pair[1] == 1])
    good = len([1 for pair in pairs if pair[0] == 1 and pair[1] == 1])
    return float(good) / expected