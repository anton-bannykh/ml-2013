import numpy as np
import scipy.optimize as spopt

factors = 10
maxIters = 10
debugOutput = True


def learn(maxRating, nUsers, nItems, X, Y, L):
    userRates, itemRates = getRates(nUsers, nItems, X, Y)
    p = np.random.random((nUsers, factors))
    q = np.random.random((nItems, factors))

    for it in range(maxIters):
        if debugOutput:
            print("Step {0}:".format(it))
        # P step
        for user in range(nUsers):
            p[user] = ridgeRegression1(user, q, p[user], userRates[user], L)
        # Q step
        for item in range(nItems):
            q[item] = ridgeRegression1(item, p, q[item], itemRates[item], L)

    return lambda t: np.inner(p[t[0]], q[t[1]])


def ridgeRegression1(user, q, init, userRates, L, cycles=1):
    if len(userRates) == 0:
        return init

    X = np.array([q[item] for item, rating in userRates])
    Y = np.array([rating for item, rating in userRates])
    error = np.array([y - np.inner(init, x) for x, y in zip(X, Y)])
    for cycle in range(cycles):
        for k in range(factors):
            for i in range(len(X)):
                error[i] += X[i][k] * init[k]

            init[k] = 0
            a, d = 0, 0
            for i in range(len(X)):
                a += X[i][k] * X[i][k]
                d += X[i][k] * error[i]

            init[k] = d / (L + a)
            for i in range(len(X)):
                error[i] -= X[i][k] * init[k]

    return init


def getRates(nUsers, nItems, X, Y):
    userRates, itemRates = [[] for dummy in range(nUsers)], [[] for dummy in range(nItems)]

    for (user, item), rating in zip(X, Y):
        userRates[user].append((item, rating))
        itemRates[item].append((user, rating))

    return userRates, itemRates