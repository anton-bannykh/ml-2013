import numpy
from numpy.linalg import norm
from math import exp
from random import seed, randint

from common import *

seed('kernel svm')

gaussianKernel = lambda x, y: exp(-norm(x - y) ** 2)
polynomialKernel = lambda x, y: (numpy.dot(x, y) + 1) ** 2

# Уныние. Перевод с псевдокода на питон :(
def mkKernelSVMClassifier(C, K):
    def train(X, y, iterations=20, eps=1e-5):
        n, m = X.shape
        a = numpy.zeros(m)
        b = 0
        def f(x):
            return sum(ai * yi * K(xi, x) for ai, yi, xi in zip(a, y, X)) + b
        it, it2 = 0, 0
        while it < iterations and it2 < 100:
            changed = False
            for i in range(m):
                Ei = f(X[i]) - y[i]
                if (y[i] * Ei < -eps and a[i] < C) or (y[i] * Ei > eps and a[i] > 0):
                    j = randint(0, m - 1)
                    while i == j:
                        j = randint(0, m - 1)
                    Ej = f(X[j]) - y[j]
                    oldAi, oldAj = a[i], a[j]
                    if y[i] != y[j]:
                        L, H = max(0, a[j] - a[i]), min(C, C + a[j] - a[i])
                    else:
                        L, H = max(0, a[i] + a[j] - C), min(C, a[i] + a[j])
                    if L == H:
                        continue
                    Kij, Kii, Kjj = K(X[i], X[j]), K(X[i], X[i]), K(X[j], X[j])
                    eta = 2 * Kij - Kii - Kjj
                    if eta >= 0:
                        continue
                    a[j] -= y[j] * (Ei - Ej) / eta
                    a[j] = max(L, min(H, a[j]))
                    if abs(a[j] - oldAj) < eps:
                        continue
                    a[i] += y[i] * y[j] * (oldAj - a[j])
                    b1 = b - Ei - y[i] * (a[i] - oldAi) * Kii - y[j] * (a[j] - oldAj) * Kij
                    b2 = b - Ej - y[i] * (a[i] - oldAi) * Kij - y[j] * (a[j] - oldAj) * Kjj
                    if 0 < a[i] < C:
                        b = b1
                    elif 0 < a[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    changed = True
            if not changed:
                it += 1
            else:
                it = 0
            it2 += 1
        return lambda x : numpy.sign(f(x))
    return train

X, y = loadData()

bestC, bestCF1, bestRes = None, None, None

for C in 2. ** numpy.arange(-5, 20):
    XStudy, yStudy, XTest, yTest = split(X, y, int(0.5 * y.size))
    classifier = mkKernelSVMClassifier(C, polynomialKernel)(XStudy, yStudy, C)
    res = checkClassifier(classifier, XTest, yTest)
    f1Here = f1(res)
    print ("SMV with C=%f, F1=%f"%(C, f1Here))
    if not bestCF1 or f1Here > bestCF1:
        bestCF1, bestC, bestRes = f1Here, C, res

for (w, f) in [('accurancy', accurancy), ('precision', precision), ('recall', recall), ('F1 measure', f1)]:
    print ("SVM with C=%f %s: %f"%(bestC, w, f(bestRes)))
