import errors
from time import time
from learning import learn


dataFolder = "itmo-recsys-data/"
setsN = 1
setFiles = range(1, setsN + 1)


def main():
    for i in setFiles:
        setName = dataFolder + "movielensfold{0}.txt".format(i)
        ansName = dataFolder + "movielensfold{0}ans.txt".format(i)
        #setName = dataFolder + "test{0}.txt".format(i)
        #ansName = dataFolder + "test{0}ans.txt".format(i)
        solveOne(i, setName, ansName)


def solveOne(setN, setName, ansName):
    startTime = time()
    maxRating, nUsers, nItems, trainX, trainY, testX, testY = loadData(setName, ansName)

    print("Set #{0}: read {1} users, {2} items, {3} train set ratings and {4} test set ratings".format(setN, nUsers,
                                                                                                       nItems,
                                                                                                       len(trainX),
                                                                                                       len(testX)))
    predictedRs = learn(maxRating, nUsers, nItems, trainX, trainY, testX)
    print("Set #{0}: RMSE = {1}".format(setN, errors.rmse(predictedRs, testY)))
    endTime = time()
    print("Set #{0}: Leaning done in {1} seconds".format(setN, endTime - startTime))


def loadData(setName, ansName):
    lines = []
    for line in open(setName, "r"):
        lines.append(line)

    maxRating, nUsers, nItems, nTrain, nTest = map(int, lines[0].split())

    trainX, trainY = [], []
    testX, testY = [], []

    for i in range(nTrain):
        user, item, rating = map(int, lines[i + 1].split())
        trainX.append((user, item))
        trainY.append(rating)

    for i in range(nTest):
        user, item = map(int, lines[1 + nTrain + i].split())
        testX.append((user, item))

    for line in open(ansName, "r"):
        testY.append(int(line))

    return maxRating, nUsers, nItems, trainX, trainY, testX, testY


main()