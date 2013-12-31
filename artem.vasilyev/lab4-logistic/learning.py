import matplotlib.pyplot as pyplot
import numpy as np
import logistic

positiveLabel, negativeLabel = 1, 0

train_rate = 0.6
crossval_rate = 0.2
assert (train_rate + crossval_rate < 1)

maxL = 15
lRange = 2.0 ** np.arange(-maxL, maxL + 1)

def learn(X, Y):
    assert (len(X) == len(Y))
    scaleFeatures(X)
    prependOnes(X)

    trainSize = int(train_rate * len(X))
    crossvalSize = int(crossval_rate * len(X))
    testSize = len(X) - trainSize - crossvalSize

    trainX, trainY = X[:trainSize], Y[:trainSize]
    crossvalX, crossvalY = X[trainSize: trainSize + crossvalSize], Y[trainSize: trainSize + crossvalSize]
    testX, testY = X[-testSize:], Y[-testSize:]

    classifiers = [logistic.learn(trainX, trainY, l) for l in lRange]

    trainErrors = [error(trainX, trainY, classifier) for classifier in classifiers]
    crossvalErrors = [error(crossvalX, crossvalY, classifier) for classifier in classifiers]

    plotErrors(lRange, trainErrors, crossvalErrors)

    bestError, bestC, bestClassifier = min(zip(crossvalErrors, lRange, classifiers))

    return bestClassifier, bestC, error(testX, testY, bestClassifier), f1score(testX, testY, bestClassifier)


def scaleFeatures(X):
    mean = sum(X) / len(X)
    stddev = sum((x - mean) * (x - mean) for x in X) / len(X)
    stddev = np.sqrt(stddev)

    for i in range(len(X)):
        X[i] -= mean
        X[i] /= stddev


def prependOnes(X):
    for i in range(len(X)):
        X[i] = np.insert(X[i], 0, 1)


def error(X, Y, classifier):
    assert (len(X) == len(Y))
    errors = [(classifier(x) != y) for x, y in zip(X, Y)].count(True)
    return float(errors) / len(X)


def f1score(X, Y, classifier):
    assert (len(X) == len(Y))
    predicted = list(map(classifier, X))
    pairs = list(zip(Y, predicted))

    tp = pairs.count((positiveLabel, positiveLabel))
    fp = pairs.count((positiveLabel, negativeLabel))
    fn = pairs.count((negativeLabel, positiveLabel))

    precision, recall, f1 = 0, 0, 0
    if tp + fp > 0:
        precision = float(tp) / (tp + fp)
    if tp + fn > 0:
        recall = float(tp) / (tp + fn)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def plotErrors(cRange, trainErrors, crossvalErrors):
    with pyplot.xkcd():
        pyplot.plot(cRange, trainErrors, label='Training set error')
        pyplot.plot(cRange, crossvalErrors, label='Cross-validation set error')
        pyplot.legend()
        pyplot.xscale('log')
        pyplot.xlabel('L')
        pyplot.ylabel('Error')
        pyplot.show()