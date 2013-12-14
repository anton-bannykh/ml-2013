import numpy as np
import scipy.optimize as spopt


factors = 50
maxIters = 100
debugOutput = False


def learn(maxRating, nUsers, nItems, X, Y, L):
    dim = nUsers * factors + nItems * factors
    args = (maxRating, nUsers, nItems, X, Y, L)

    x0 = np.random.random(dim)
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


def f(theta, *args):
    (maxRating, nUsers, nItems, X, Y, L) = args
    p, q = unpack(theta, nUsers, nItems)

    trainError, regError = 0, 0
    for (user, item), rating in zip(X, Y):
        predicted = np.inner(p[user], q[item])
        trainError += (predicted - rating) ** 2

    regError = L * (sum(np.inner(p1, p1) for p1 in p) + sum(np.inner(q1, q1) for q1 in q))
    return (trainError + regError) / 2


def fPrime(theta, *args):
    (maxRating, nUsers, nItems, X, Y, L) = args
    p, q = unpack(theta, nUsers, nItems)

    grad = theta * L

    gradU = np.zeros((nUsers, factors))
    gradI = np.zeros((nItems, factors))

    for (user, item), rating in zip(X, Y):
        predicted = np.inner(p[user], q[item])
        gradU[user] += q[item] * (predicted - rating)
        gradI[item] += p[user] * (predicted - rating)

    for i in range(nUsers):
        grad[i * factors:(i + 1) * factors] += gradU[i]

    for i in range(nItems):
        grad[(nUsers + i) * factors:(nUsers + i + 1) * factors] += gradI[i]

    return grad
