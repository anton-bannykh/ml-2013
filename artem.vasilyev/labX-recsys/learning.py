import numpy as np
import matplotlib.pyplot as pyplot
import errors

trainRatio = 0.8
maxL = 5
lRange = 2.0 ** np.arange(-maxL, maxL + 1)
methodName = 'als1'


def learn(maxRating, nUsers, nItems, X, Y, testX):
    assert (len(X) == len(Y))
    movieMean = centerMean(X, Y, nItems)

    trainSize = int(len(X) * trainRatio)
    trainX, trainY = X[:trainSize], Y[:trainSize]
    crossvalX, crossvalY = X[trainSize:], Y[trainSize:]

    learningMethod = methodByName(methodName)
    recs = [learningMethod(maxRating, nUsers, nItems, trainX, trainY, L) for L in lRange]

    trainErrors = [errors.rmse(recommend(rec, trainX), trainY) for rec in recs]
    crossvalErrors = [errors.rmse(recommend(rec, crossvalX), crossvalY) for rec in recs]

    plotErrors(lRange, trainErrors, crossvalErrors)
    bestError, bestL, bestRec = min(zip(crossvalErrors, lRange, recs))

    testY = recommend(bestRec, testX)
    for i in range(len(testX)):
        testY[i] += movieMean[testX[i][1]]

    return bestL, testY


def methodByName(name):
    if name == 'svd':
        import svd
        return svd.learn
    elif name == 'als':
        import als
        return als.learn
    elif name == 'als1':
        import als1
        return als1.learn
    return None


def recommend(rec, X):
    return list(map(rec, X))


def centerMean(X, Y, nItems):
    sum, count = np.zeros(nItems), np.zeros(nItems)

    for (_, item), rating in zip(X, Y):
        sum[item] += rating
        count[item] += 1

    for i in range(nItems):
        if count[i] == 0:
            count[i] = 1

    mean = sum / count

    for i in range(len(Y)):
        Y[i] -= mean[X[i][1]]
    return mean


def plotErrors(cRange, trainErrors, crossvalErrors):
    pyplot.plot(cRange, trainErrors, label='Training set error')
    pyplot.plot(cRange, crossvalErrors, label='Cross-validation set error')
    pyplot.legend()
    pyplot.xscale('log')
    pyplot.xlabel('L')
    pyplot.ylabel('Error')
    pyplot.show()