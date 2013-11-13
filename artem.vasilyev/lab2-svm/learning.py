import matplotlib.pyplot as pyplot
import numpy as np
import svm

train_rate = 0.6
crossval_rate = 0.2
assert (train_rate + crossval_rate < 1)

maxC = 15
cRange = 2.0 ** np.arange(-maxC, maxC + 1)

def learn(X, Y):
    assert (len(X) == len(Y))

    trainSize = int(train_rate * len(X))
    crossvalSize = int(crossval_rate * len(X))
    testSize = len(X) - trainSize - crossvalSize

    trainX, trainY = X[:trainSize], Y[:trainSize]
    crossvalX, crossvalY = X[trainSize: trainSize + crossvalSize], Y[trainSize: trainSize + crossvalSize]
    testX, testY = X[-testSize:], Y[-testSize:]

    classifiers = [svm.learn(trainX, trainY, c) for c in cRange]

    trainErrors = [error(trainX, trainY, classifier) for classifier in classifiers]
    crossvalErrors = [error(crossvalX, crossvalY, classifier) for classifier in classifiers]

    print(cRange)
    print(trainErrors)
    print(crossvalErrors)
    plotErrors(cRange, trainErrors, crossvalErrors)

    bestError, bestC, bestClassifier = min(zip(crossvalErrors, cRange, classifiers))
    return bestClassifier, bestC, error(testX, testY, bestClassifier), f1score(testX, testY, bestClassifier)

def error(X, Y, classifier):
    assert (len(X) == len(Y))
    errors = [(classifier(x) * y > 0) for x, y in zip(X, Y)].count(False)
    return float(errors) / len(X)

def f1score(X, Y, classifier):
    assert (len(X) == len(Y))
    predicted = list(map(classifier, X))
    pairs = list(zip(Y, predicted))

    tp = pairs.count((1, 1))
    fp = pairs.count((1, -1))
    fn = pairs.count((-1, 1))

    precision, recall, f1 = 0, 0, 0
    if tp + fp > 0:
        precision = float(tp) / (tp + fp)
    if tp + fn > 0:
        recall = float(tp) / (tp + fn)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def plotErrors(cRange, trainErrors, crossvalErrors):
    pyplot.plot(cRange, trainErrors, label='Training set error')
    pyplot.plot(cRange, crossvalErrors, label='Cross-validation set error')
    pyplot.legend()
    pyplot.xscale('log')
    pyplot.xlabel('C')
    pyplot.ylabel('Error')
    pyplot.show()
    return
