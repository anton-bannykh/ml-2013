from cvxopt import matrix
import random as rd


class Smo:
    def __init__(self, c, vectors, classes, tolerance, kernel):
        m = len(vectors)
        kernel_matrix = matrix(0., (m, m))
        for i in range(0, m):
            for j in range(0, m):
                kernel_matrix[i, j] = kernel(vectors[i], vectors[j])
        self.kernel_matrix = kernel_matrix
        self.kernel = kernel
        self.m = m
        self.vectors = vectors
        self.y = classes
        self.alphas = [0] * m
        self.b = 0
        self.tolerance = tolerance
        self.c = c
        self.eps = 1e-5
        self.tune()
        self.b = self.calculate_b()

    def tune(self):
        passes = 0
        count = 0
        while passes < 4:
            changed_alphas = False
            count += 1
            if count % 100 == 0:
                print("100 iterations passed")
                print(self.optimized_function())
            for p in range(0, self.m):
                if count % 2 == 0 and (self.alphas[p] < 1e-8 or self.alphas[p] > self.c - 1e-8):
                    continue
                q = rd.randint(0, self.m - 1)
                while p == q:
                    q = rd.randint(0, self.m - 1)
                found = (self.f(p) - self.y[p] + self.tolerance < self.f(q) - self.y[q]
                         - self.tolerance) and self.is_correct(p, +1) and self.is_correct(q, -1)
                if not found:
                    continue
                changed_alphas = True
                eta = ((self.f(q) - self.y[q]) - (self.f(p) - self.y[p])) / (
                    self.kernel_matrix[p, p] - 2 * self.kernel_matrix[p, q] + self.kernel_matrix[q, q])
                eta = self.correct_eta(eta, p, +1)
                eta = self.correct_eta(eta, q, -1)
                self.alphas[p] += eta * self.y[p]
                self.alphas[q] -= eta * self.y[q]
            if not changed_alphas:
                passes += 1
            else:
                passes = 0

    def is_correct(self, i, sign):
        return (self.alphas[i] < self.c and self.y[i] == sign) or (self.alphas[i] > 0 and self.y[i] == -sign)

    def correct_eta(self, eta, i, sign):
        result = eta
        if self.alphas[i] + sign * self.y[i] * eta > self.c:
            result = (self.c - self.alphas[i]) / (self.y[i] * sign)
        if self.alphas[i] + sign * self.y[i] * eta < 0:
            result = - self.alphas[i] / (self.y[i] * sign)
        return result

    def calculate_b(self):
        max = self.f(0)
        for i in range(1, self.m):
            if self.y[i] == 1 and self.alphas[i] > self.eps:
                val = self.f(i)
                if val > max:
                    max = val
        min = self.f(0)
        for i in range(1, self.m):
            if self.y[i] == -1 and self.alphas[i] > self.eps:
                val = self.f(i)
                if val < min:
                    min = val
        return -(max + min) / 2

    def f(self, i):
        result = self.b
        for j in range(0, self.m):
            result += self.y[j] * self.alphas[j] * self.kernel_matrix[i, j]
        return result

    def real_f(self, x):
        result = self.b
        for vector, clazz, alpha in zip(self.vectors, self.y, self.alphas):
            result += clazz * alpha * self.kernel(x, vector)
        return result

    def optimized_function(self):
        result = 0
        for alpha in self.alphas:
            result += alpha
        for i in range(0, self.m):
            for j in range(0, self.m):
                result -= 0.5 * self.y[i] * self.y[j] * self.alphas[i] * self.alphas[j] * self.kernel_matrix[i, j]
        return result

    def classify(self, x):
        res = self.real_f(x)
        if res > 0:
            return 1
        else:
            return -1