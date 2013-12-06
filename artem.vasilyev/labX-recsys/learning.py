import numpy as np
import matplotlib.pyplot as pyplot
import svd
import errors

trainRatio = 0.75
maxL = 3
lRange = 2.0 ** np.arange(-maxL, maxL + 1)


def learn(maxRating, nUsers, nItems, X, Y, testX):
    assert (len(X) == len(Y))
    movieMean = centerMean(X, Y, nItems)

    trainSize = int(len(X) * trainRatio)
    trainX, trainY = X[:trainSize], Y[:trainSize]
    crossvalX, crossvalY = X[trainSize:], Y[trainSize:]

    #print(X, Y)

    #print(trainX, trainY)
    #print(crossvalX, crossvalY)

    recs = [svd.learn(maxRating, nUsers, nItems, trainX, trainY, L) for L in lRange]

    trainErrors = [errors.rmse(recommend(rec, trainX), trainY) for rec in recs]
    crossvalErrors = [errors.rmse(recommend(rec, crossvalX), crossvalY) for rec in recs]

    plotErrors(lRange, trainErrors, crossvalErrors)
    bestError, bestC, bestRec = min(zip(crossvalErrors, lRange, recs))

    #print(recommend(bestRec, trainX))

    testY = list(recommend(bestRec, testX))
    #print(testY)
    for i in range(len(testX)):
        testY[i] += movieMean[testX[i][1]]

    return testY


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