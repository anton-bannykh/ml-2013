from cvxopt import solvers
from numpy import array
from cvxopt import matrix
import numpy as np


class Svm:
    def __init__(self, c, vectors, classes, eps=1e-8):
        alphas = solvers.qp(*self.get_qp_params(vectors, classes, c))["x"]
        support_vectors = []
        for i in range(len(vectors)):
            if alphas[i] > eps:
                support_vectors.append((alphas[i], vectors[i], classes[i]))

        n = len(vectors[0])
        w = np.zeros(n)
        for alpha, x, y in support_vectors:
            for i in range(n):
                w[i] += alpha * y * x[i]

        count = 0
        w0 = 0
        for alpha, x, y in support_vectors:
            w0 += array(w).dot(x) - y
            count += 1
        w0 /= count
        self.w0 = w0
        self.w = w
        self.support_vectors = support_vectors

    @staticmethod
    def kernel(a, b):
        return array(a).dot(array(b))

    @staticmethod
    def get_qp_params(vectors, classes, c):
        n = len(vectors)
        p = matrix(0., (n, n))
        for i in range(n):
            for j in range(n):
                p[i, j] = classes[i] * classes[j] * Svm.kernel(vectors[i], vectors[j])
        q = matrix(-1., (n, 1))
        g = matrix(0., (2 * n, n))
        h = matrix(0., (2 * n, 1))
        for i in range(n):
            g[2 * i, i] = 1
            h[2 * i, 0] = c
            g[2 * i + 1, i] = -1
        a = matrix(classes, (1, n), "d")
        b = matrix(0., (1, 1))
        return p, q, g, h, a, b


    def classify(self, x):
        res = array(x).dot(self.w) - self.w0
        if res > 0:
            return 1
        else:
            return -1