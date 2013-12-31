# coding=utf-8
import datetime
import numpy as np
from numpy.random import random as randvec
from itertools import product

def RMSE(data, predictor):
    predictor = predictor or (lambda u, i: 0)
    predictions = np.array([predictor(u, i) for u, i, x in data])
    answers = data[:, 2]
    return np.sqrt(np.mean((answers - predictions) ** 2))

TRAIN_FILE_NAME = 'itmo-recsys-data/movielensfold%d.txt'
TEST_FILE_NAME = 'itmo-recsys-data/movielensfold%dans.txt'


def get_ground_truth(n):
    print '[INFO] Getting ground trurh for dataset #d...' % n
    train_file = open(TRAIN_FILE_NAME% n)
    test_file = open(TEST_FILE_NAME % n)
    parse_line = lambda: map(int, train_file.readline().split())
    x, users, items, ntrain, ntest = parse_line()
    train_data, test_data = np.empty((ntrain, 3)), np.empty((ntest, 3))
    for i in range(ntrain):
        train_data[i] = parse_line()
    for i in range(ntest):
        test_data[i][:2] = parse_line()
    for i in range(ntest):
        test_data[i][2] = int(test_file.readline())
    print '[INFO] Done.'
    return train_data, test_data, users, items


def get_normalized_info(train_set, users, items):
    print '[INFO] Normalization...'
    m = np.average(train_set[:, 2])
    users_count, b_user = np.zeros(users), np.zeros(users)
    items_count, b_item = np.zeros(items), np.zeros(items)
    for u, i, r in train_set:
        users_count[u] += 1
        items_count[i] += 1
        b_user[u] += r - m
        b_item[i] += r - m
    for i in range(items):
        b_item[i] = b_item[i] / items_count[i] if items_count[i] != 0 else 0
    for u in range(users):
        b_user[u] = b_user[u] / users_count[u] if users_count[u] != 0 else 0
    print '[INFO] Done.'
    return m, b_user, b_item


def sgd(data, users, itesm, lam1, lam2, gamma1, gamma2):
    print '[INFO] Start learning on λ1 = %.6f, λ2 = %.6f, ɣ1 = %.6f, ɣ2 = %.6f' % (lam1, lam2, gamma1, gamma2)
    mu, bu, bi = get_normalized_info(data, users, itesm)
    p, q = randvec((users, 50)) - 1/2., randvec((itesm, 50)) - 1/2.
    steps = 10
    for step in range(steps):
        print '[INFO] Step %d/%d...' % (step + 1, steps)
        for u, i, r in data:
            e = r - (mu + bu[u] + bi[i] + np.dot(p[u], q[i]))
            bu[u] += gamma1 * (e - lam1 * bu[u])
            bi[i] += gamma1 * (e - lam1 * bi[i])
            pu1 = p[u] + gamma2 * (e * q[i] - lam2 * p[u])
            qi1 = q[i] + gamma2 * (e * p[u] - lam2 * q[i])
            p[u], q[i] = pu1, qi1
    print '[INFO] Learned.'
    return lambda u, i: mu + bu[u] + bi[i] + np.dot(p[u], q[i])


def regularize(train_set, validation_set, user_len, item_len):
    print '[INFO] Start regularization...'
    rmse_best = 1
    lam1_best, lam2_best, g1_best, g2_best = 0, 0, 0, 0
    predictor_best = None
    for gamma1, gamma2 in [(0.0013, 0.0013)]:#product(0.33 ** np.array([5.5, 6.0, 6.5]), repeat=2):
        for lam1, lam2 in [(0.5, 0.005), (1, 0.01)]:#product([1, 0.01, 0.02, 0.008], repeat=2):
            predictor = sgd(train_set, user_len, item_len, lam1, lam2, gamma1, gamma2)
            rmse = RMSE(validation_set, predictor)
            print '[INFO] rmse = %.6f' % rmse
            if rmse < rmse_best:
                print '[INFO] Updating rmse and predictor.'
                rmse_best = rmse
                lam1_best, lam2_best, g1_best, g2_best = lam1, lam2, gamma1, gamma2
                predictor_best = predictor
    print '[INFO] Regularization done.'
    return lam1_best, lam2_best, g1_best, g2_best, predictor_best


for n in xrange(1, 6):
    train_data, test_data, user_len, item_len = get_ground_truth(n)
    split = int(len(train_data) * 0.2)
    train_set, validation_set = train_data[split:], train_data[:split]
    l1, l2, g1, g2, predictor = regularize(train_set, validation_set, user_len, item_len)
    rmse = RMSE(test_data, predictor)
    print 'dataset #%d, rmse: %.6f, λ1 = %.6f, λ2 = %.6f, ɣ1 = %.6f, ɣ2 = %.6f' % (n, rmse, l1, l2, g1, g2)


# dataset #1, rmse: 0.915550, λ1 = 0.050000, λ2 = 0.050000, ɣ1 = 0.005000, ɣ2 = 0.005000
# dataset #2, rmse: 0.916771, λ1 = 0.050000, λ2 = 0.050000, ɣ1 = 0.005000, ɣ2 = 0.005000
# dataset #3, rmse: 0.913951, λ1 = 0.050000, λ2 = 0.050000, ɣ1 = 0.005000, ɣ2 = 0.005000
# dataset #4, rmse: 0.922094, λ1 = 0.050000, λ2 = 0.050000, ɣ1 = 0.005000, ɣ2 = 0.005000
# dataset #5, rmse: 0.916731, λ1 = 0.050000, λ2 = 0.050000, ɣ1 = 0.005000, ɣ2 = 0.005000