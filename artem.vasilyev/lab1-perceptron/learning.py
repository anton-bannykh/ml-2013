import matplotlib.pyplot as pyplot
import perceptron

max_iters = 100
train_rate = 0.6
crossval_rate = 0.2
assert (train_rate + crossval_rate < 1)


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


def learn(X, Y):
    assert (len(X) == len(Y))

    trainSize = int(train_rate * len(X))
    crossvalSize = int(crossval_rate * len(X))
    testSize = len(X) - trainSize - crossvalSize

    trainX, trainY = X[:trainSize], Y[:trainSize]
    crossvalX, crossvalY = X[trainSize: trainSize + crossvalSize], Y[trainSize: trainSize + crossvalSize]
    testX, testY = X[-testSize:], Y[-testSize:]

    check_iters = range(1, max_iters + 1)
    classifiers = perceptron.learnAll(trainX, trainY, max_iters)
    trainErrors = [error(trainX, trainY, classifier) for classifier in classifiers]
    crossvalErrors = [error(crossvalX, crossvalY, classifier) for classifier in classifiers]

    plotErrors(check_iters, trainErrors, crossvalErrors)

    bestError, bestIters = min(zip(crossvalErrors, check_iters))
    bestClassifier = perceptron.learn(trainX, trainY, bestIters)

    return bestClassifier, bestIters, error(testX, testY, bestClassifier), f1score(testX, testY, bestClassifier)


def plotErrors(check_iters, trainErrors, crossvalErrors):
    with pyplot.xkcd():
        pyplot.plot(check_iters, trainErrors, label='Training set error')
        pyplot.plot(check_iters, crossvalErrors, label='Cross-validation set error')
        pyplot.legend()
        pyplot.xlabel('Iterations')
        pyplot.ylabel('Error')
        pyplot.show()
    return
