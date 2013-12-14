import numpy as np

factors = 30
maxIters = 10
debugOutput = False


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


def ridgeRegression1(user, q, init, userRates, L):
    if len(userRates) == 0:
        return init

    X = np.array([q[item] for item, rating in userRates]).T
    error = np.array([rating - np.inner(init, q[item]) for item, rating in userRates])
    precalcA = np.array([np.sum(x ** 2) for x in X])

    for k in range(factors):
        error += X[k] * init[k]

        init[k] = 0
        a, d = precalcA[k], sum(X[k] * error)

        init[k] = d / (L * len(userRates) + a)
        error -= X[k] * init[k]

    return init


def getRates(nUsers, nItems, X, Y):
    userRates, itemRates = [[] for dummy in range(nUsers)], [[] for dummy in range(nItems)]

    for (user, item), rating in zip(X, Y):
        userRates[user].append((item, rating))
        itemRates[item].append((user, rating))

    return userRates, itemRates