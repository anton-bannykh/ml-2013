import numpy as np
import scipy.optimize as spopt

maxIters = 500
debugOutput = True

def learn(X, Y, L):
    assert len(X) == len(Y)

    dimX = len(X[0])
    args = (dimX, X, Y, L)

    x0 = np.zeros(dimX)
    #theta = spopt.fmin_cg(f, x0, fprime=fPrime, args=args, maxiter=maxIters, disp=debugOutput)
    theta = spopt.fmin_bfgs(f, x0, fprime=fPrime, args=args, maxiter=maxIters, disp=debugOutput)

    return lambda x: classify(x, theta.copy())


def f(theta, *args):
    dimX, X, Y, L = args
    trainError = error(X, Y, theta)
    regError = L * 0.5 * (np.inner(theta, theta) - theta[0] * theta[0])
    return (trainError + regError) / len(X)


def fPrime(theta, *args):
    dimX, X, Y, L = args
    grad = theta * L
    grad[0] = 0
    for x, y in zip(X, Y):
        pr = prob(x, theta)
        grad += (pr - y) * x
    return grad / len(X)


def error(X, Y, theta):
    probs = [prob(x, theta) for x in X]
    errors = [-np.log(pr) if y == 1 else -np.log(1 - pr) for y, pr in zip(Y, probs)]
    return sum(errors)


def prob(x, theta):
    return sigmoid(np.inner(x, theta))


def sigmoid(x):
    if x < -200:
        return 0
    if x > 200:
        return 1
    return 1.0 / (1 + np.exp(-x))


def classify(x, theta):
    val = prob(x, theta)
    return 1 if val >= 0.5 else 0
