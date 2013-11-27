from mlaux import *
import numpy as np
import random
import numpy
from math import exp


gaussian = lambda x, y: exp(-np.sum((x - y) ** 2) * 1e-6 / 2)
polynomial = lambda x, y: float(1 + np.dot(x, y)) ** 3
scalar = lambda x, y: float(np.dot(x, y))

def train(data, C, K, eps=1e-7):
    minmax = lambda p, lo, hi: lo if p < lo else (hi if p > hi else p)

    x, y = split_xy(data)
    n, d = x.shape
    A = np.zeros(n)
    b = 0

    k = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            k[i][j] = K(x[i], x[j])

    F = lambda i: b + np.sum(np.dot(k[i], A * y))

    has_changes = True
    itercount = 0
    while has_changes and itercount < 10e5:
        has_changes = False
        for i in range(n):
            Ei = F(i) - y[i]
            if (y[i] * Ei < -eps and A[i] < C) or (y[i] * Ei > eps and A[i] > 0):
                j = random.randint(0, n - 2)
                while j == i:
                    j = random.randint(0, n - 2)
                Ej = F(j) - y[j]
                old_Ai, old_Aj = A[i], A[j]
                if y[i] != y[j]:
                    L, H = max(0, old_Aj - old_Ai), min(C, C + old_Aj - old_Ai)
                else:
                    L, H = max(0, old_Ai + old_Aj - C), min(C, old_Ai + old_Aj)
                if L == H:
                    continue
                eta = 2 * k[i][j] - k[i][i] - k[j][j]
                if eta > 0:
                    continue
                A[j] = minmax(A[j] - y[j] * (Ei - Ej) / eta, L, H)
                if abs(A[j] - old_Aj) < 1e-5:
                    continue

                A[i] = A[i] + y[i] * y[j] * (old_Aj - A[j])
                b1 = b - Ei - y[i] * (A[i] - old_Ai) * K(x[i], x[i]) - y[j] * (A[j] - old_Aj) * K(x[i], x[j])
                b2 = b - Ej - y[i] * (A[i] - old_Ai) * K(x[i], x[j]) - y[j] * (A[j] - old_Aj) * K(x[j], x[j])
                if 0 < A[i] < C:
                    b = b1
                elif 0 < A[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                has_changes = True
                itercount += 1

    return A, b


def test(data_train, data, p, kernel):
    x, y = split_xy(data_train)
    x_test, test_Y = split_xy(data)
    n, n_test = len(y), len(test_Y)
    a, b = p
    stat = Statistic()
    f = lambda v: b + np.sum(a * y * np.apply_along_axis(lambda w: kernel(v, w), 1, x))
    for i in range(n_test):
        yc = f(x_test[i])
        if yc > 0 and test_Y[i] == 1:
            stat.true_pos += 1
        elif yc > 0 and test_Y[i] == -1:
            stat.false_pos += 1
        elif yc < 0 and test_Y[i] == 1:
            stat.false_neg += 1
        else:
            stat.true_neg += 1
    return stat


data = grouped(load_data())
data_train, data_test = split(data)
best_reg_const = 200.0
best_f1 = 0

# teaching on polynomial kernel
for C in xrange(450, 550, 5):
    print 'try C = %f' % C
    alpha_g, b_g = train(data_train, C, polynomial)
    stat = test(data_train, data_test, (alpha_g, b_g), polynomial)
    try:
        f1 = stat.f1()
    except ZeroDivisionError:
        continue
    if f1 > best_f1:
        best_f1 = f1
        best_reg_const = C

reg_const = best_reg_const
print best_reg_const


def run(data_train, reg_const, kernel, message):
    alpha_g, b_g = train(data_train, reg_const, kernel)
    stat = test(data_train, data_test, (alpha_g, b_g), kernel)
    print("SMO with %s kernel:" % message)
    print("\tprecision: %.2f\n\trecall: %.2f\n\terror: %.2f\n\tF1: %.2f\n" %
          (stat.precision(), stat.recall(), stat.error(), stat.f1()))

run(data_train, reg_const, polynomial, "polynomial")
run(data_train, reg_const, gaussian, "gaussian")