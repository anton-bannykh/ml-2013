import numpy as np
import scipy.optimize as spopt


factors = 30
maxIters = 30
debugOutput = True


def learn(maxRating, nUsers, nItems, X, Y, L):
    dim = nUsers * factors + nItems * factors
    args = (maxRating, nUsers, nItems, X, Y, L)

    if debugOutput:
        print("Learning for L = {0}".format(L))

    thetaU0, thetaI0 = np.random.random((nUsers, factors)), np.random.random((nItems, factors))
    for u in range(nUsers):
        thetaU0[u][0] = 1
    for i in range(nItems):
        thetaI0[i][1] = 1
    x0 = pack(thetaU0, thetaI0)
    #theta = spopt.fmin_bfgs(f, x0, fprime=fPrime, args=args, maxiter=maxIters, disp=debugOutput)
    #theta = spopt.fmin_cg(f, x0, fprime=fPrime, args=args, maxiter=maxIters, disp=debugOutput)
    theta, opt, dct = spopt.fmin_l_bfgs_b(f, x0, fprime=fPrime, args=args, maxiter=maxIters, disp=debugOutput, m=100)

    p, q = unpack(theta, nUsers, nItems)
    return lambda t: np.inner(p[t[0]], q[t[1]])


def unpack(theta, nUsers, nItems):
    p, q = np.empty((nUsers, factors)), np.empty((nItems, factors))
    for i in range(nUsers):
        p[i] = np.array(theta[i * factors:(i + 1) * factors])
    for i in range(nItems):
        q[i] = np.array(theta[(nUsers + i) * factors:(nUsers + i + 1) * factors])
    return p, q


def pack(thetaU, thetaI):
    grad = np.zeros(factors * (len(thetaU) + len(thetaI)))
    nUsers, nItems = len(thetaU), len(thetaI)
    for u in range(len(thetaU)):
        grad[u * factors:(u + 1) * factors] += thetaU[u]
    for i in range(len(thetaI)):
        grad[(nUsers + i) * factors:(nUsers + i + 1) * factors] += thetaI[i]
    return grad


def f(theta, *args):
    (maxRating, nUsers, nItems, X, Y, L) = args
    p, q = unpack(theta, nUsers, nItems)

    trainError, regError = 0, 0
    for (user, item), rating in zip(X, Y):
        predicted = np.inner(p[user], q[item])
        trainError += (predicted - rating) ** 2
        regError += L * (np.inner(p[user], p[user]) - 1 + np.inner(q[item], q[item]) - 1)

    return (trainError + regError) / 2


def fPrime(theta, *args):
    (maxRating, nUsers, nItems, X, Y, L) = args
    p, q = unpack(theta, nUsers, nItems)

    gradU = np.zeros((nUsers, factors))
    gradI = np.zeros((nItems, factors))

    for (user, item), rating in zip(X, Y):
        predicted = np.inner(p[user], q[item])
        gradU[user] += q[item] * (predicted - rating)
        gradI[item] += p[user] * (predicted - rating)

        gradU[user] += L * p[user]
        gradI[item] += L * q[item]

    for u in range(nUsers):
        gradU[u][0] = 0

    for i in range(nItems):
        gradI[i][1] = 0

    return pack(gradU, gradI)
