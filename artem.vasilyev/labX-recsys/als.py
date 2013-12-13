import numpy as np
import scipy.optimize as spopt

factors = 30
maxIters = 30
debugOutput = False


def learn(maxRating, nUsers, nItems, X, Y, L):
    userRates, itemRates = getRates(nUsers, nItems, X, Y)
    p = np.random.random((nUsers, factors))
    q = np.random.random((nItems, factors))

    for dummy in range(maxIters):
        print("Step {0}:".format(dummy))
        # P step
        for user in range(nUsers):
            p[user] = oneStep(user, q, userRates[user], L)
        # Q step
        for item in range(nItems):
            q[item] = oneStep(item, p, itemRates[item], L)

    return lambda t: np.inner(p[t[0]], q[t[1]])


def oneStep(user, q, userRates, L):
    if len(userRates) == 0:
        return np.zeros(factors)

    A = np.zeros((factors, factors))
    d = np.zeros(factors)

    for item, rating in userRates:
        qq = q[item]
        A += np.outer(qq, qq)
        d += rating * qq

    for i in range(factors):
        A[i][i] += L * len(userRates)

    try:
        result = np.linalg.solve(A, d)
        return result
    except np.linalg.LinAlgError:
        return np.zeros(factors)


def getRates(nUsers, nItems, X, Y):
    userRates, itemRates = [[] for dummy in range(nUsers)], [[] for dummy in range(nItems)]

    for (user, item), rating in zip(X, Y):
        userRates[user].append((item, rating))
        itemRates[item].append((user, rating))

    return userRates, itemRates