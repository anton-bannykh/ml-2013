import numpy as np


def classify(x, theta, theta0=0):
    val = np.inner(x, theta) + theta0
    return 1 if val >= 0 else -1


def learn(X, Y, iters):
    d = len(X[0])
    theta = np.ndarray(d, np.float64)
    theta0 = 0

    for _ in range(iters):
        for x, y in zip(X, Y):
            if y * (classify(x, theta, theta0)) < 0:
                theta += y * x
                theta0 += y

    return lambda t: classify(t, theta, theta0)


def learnAll(X, Y, iters):
    d = len(X[0])
    theta = np.ndarray(d, np.float64)
    theta0 = 0
    classifiers = []

    for _ in range(iters):
        for x, y in zip(X, Y):
            if y * (classify(x, theta, theta0)) < 0:
                theta += y * x
                theta0 += y

        classifiers.append(lambda t, theta=theta.copy(), theta0=theta0: classify(t, theta, theta0))

    return classifiers


