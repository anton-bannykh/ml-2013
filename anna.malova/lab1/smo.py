from numpy import log, exp, zeros
from numpy.linalg import norm
import numpy as np
from data import *
from random import randint

class kernelSVM:
    def __init__(self, vec, C, kernel):
        def f(cur_x):
            return sum(a_i * y_i * kernel(x_i, cur_x) for a_i, y_i, x_i in zip(self.a, y, x)) + self.b

        x = np.array([v.data for v in vec])
        y = np.array([v.label for v in vec])
        n, m = x.shape
        self.a, self.b = zeros(m), 0
        passes, max_passes = 0, 1000
        eps = 10**(-9)
        while (passes < max_passes):
            changed = False
            for i in range(m):
                e_i = f(x[i]) - y[i]
                if (y[i] * e_i < -eps and self.a[i] < C) or (y[i] * e_i > eps and self.a[i] > 0):
                    j = i
                    while j == i:
                        j = randint(0, m - 1)
                    e_j = f(x[j]) - y[j]
                    old_a = (self.a[i], self.a[j])
                    (L, H) = [(max(0, self.a[i] + self.a[j] - C), min(C, self.a[i] + self.a[j])), (max(0, self.a[j] - self.a[i]), min(C, C + self.a[j] - self.a[i]))][y[i] != y[j]]
                    if L == H:
                        continue
                    K_ij, K_ii, K_jj = kernel(x[i], x[j]), kernel(x[i], x[i]), kernel(x[j], x[j])
                    nu = 2 * K_ij - K_ii - K_jj
                    if nu >= 0:
                        continue
                    self.a[j] -= y[j] * (e_i - e_j) / nu
                    self.a[j] = H if self.a[j] > H else \
                           L if self.a[j] < L else \
                           self.a[j]
                    if abs(self.a[j] - old_a[1]) < eps:
                        continue
                    self.a[i] += y[i] * y[j] * (old_a[1] - self.a[j])
                    b1 = self.b - e_i - y[i] * (self.a[i] - old_a[0]) * K_ii - y[j] * (self.a[j] - old_a[1]) * K_ij
                    b2 = self.b - e_j - y[i] * (self.a[i] - old_a[0]) * K_ij - y[j] * (self.a[j] - old_a[1]) * K_jj
                    self.b = b1 if 0 < self.a[i] < C else \
                             b2 if 0 < self.a[j] < C else \
                             (b1 + b2) / 2
                if not changed:
                    passes += 1
                else:
                    passes = 0

        print("Training completed with C = " + str(C) + " using " + kernel.__name__)

    def get_label(self, v):
        return np.sign(self.a.dot(v.data) + self.b)

def gaussianKernel(x, y):
    γ = 1
    return exp(-γ * norm(x - y) ** 2)

def polynomialKernel(x, y):
    d = 2
    return (1 + np.dot(x, y)) ** d


def main(train_fraction):
    train, test = load_data(train_fraction)
    kernels = [gaussianKernel, polynomialKernel]

    for K in kernels:
        best_C = None
        best_score = None
        for C in 2. ** np.arange(-15, 15):
            cv_train, cv_test = split_data(train, 0.5)
            classifier = kernelSVM(cv_train, C, K)
            score = get_metrics(classifier, cv_test)['f1']
            if best_score == None or score > best_score:
                best_score = score
                print("Now best score is " + str(best_score))
                best_C = C

        classifier = kernelSVM(train, best_C, K)
        print("For " + K.__name__ + "the best C = " + str(best_C))
        print_results(classifier, test)

if __name__ == "__main__":
    main(0.7)