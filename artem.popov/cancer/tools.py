import random
import urllib2
import csv
import numpy

__author__ = 'jambo'

class DataElement:
    def __init__(self, data_list):
        self.id = int(data_list[0])
        self.answer = 1 if data_list[1] == 'M' else -1
        self.features = numpy.array([map(float, data_list[2:])])


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


def training_and_test_sets():
    """
    @rtype: list of DataElement, list of DataElement
    """
    dataset = [DataElement(list) for list in load_cancer_dataset()]
    random.shuffle(dataset)
    separate_point = int(len(dataset) * 0.8)
    return dataset[:separate_point], dataset[separate_point:]


def error_rate(answers, actual):
    pairs = zip(answers, actual)
    mistakes = len([1 for pair in pairs if pair[0] != pair[1]])
    return float(mistakes) / len(pairs)


def precision(answers, actual):
    pairs = zip(answers, actual)
    answered = len([1 for pair in pairs if pair[0] == 1])
    good = len([1 for pair in pairs if pair[0] == 1 and pair[1] == 1])
    return float(good) / answered


def recall(answers, actual):
    pairs = zip(answers, actual)
    expected = len([1 for pair in pairs if pair[1] == 1])
    good = len([1 for pair in pairs if pair[0] == 1 and pair[1] == 1])
    return float(good) / expected