import numpy as np
from numpy.random import random as randvec
from random import random

def initialize(data, lengths, factors):
    mu = np.average(data[:, 2])
    #print(mu)
    ulen, ilen = lengths[0], lengths[1]
    p, q = randvec((ulen, factors)) - 0.5, randvec((ilen, factors)) - 0.5
    nu, bu = np.zeros(ulen), np.zeros(ulen)
    ni, bi = np.zeros(ilen), np.zeros(ilen)
    for u, i, r in data:
        nu[u] += 1
        ni[i] += 1
        bu[u] += r - mu
        bi[i] += r - mu
    for i in range(ilen):
        bi[i] = bi[i] / ni[i] if ni[i] != 0 else 0
    for u in range(ulen):
        bu[u] = bu[u] / nu[u] if nu[u] != 0 else 0
    return mu, bu, bi, p, q

def learn(data, lengths, lam, gamma, iterations=10, factors=50):
    np.seterr(all='raise')
    mu, bu, bi, p, q = initialize(data, lengths, factors)
    gamma1, gamma2 = gamma
    lam1, lam2 = lam

    def predict(u, i):
        return mu + bu[u] + bi[i] + np.dot(p[u], q[i])

    for iter in range(iterations):
        print("RMSE after ", iter, " iterations: ", rmse(data, predict))
        for u, i, r in data:
            err = r - predict(u, i)
            #print("(%i, %i, %i): error = %6.2f" % (u, i, r, err))
            bu[u] += gamma1 * (err - lam1 * bu[u])
            bi[i] += gamma1 * (err - lam1 * bi[i])
            pu1 = p[u] + gamma2 * (err * q[i] - lam2 * p[u])
            qi1 = q[i] + gamma2 * (err * p[u] - lam2 * q[i])
            p[u], q[i] = pu1, qi1
            #print(bu[u], bi[i], p[u], q[i])
            #print(bu1, bi1, p1, q1)

    return predict

def rmse(data, predict):
    predictions = np.array([predict(u, i) for u, i, _ in data])
    answers = data[:, 2]
    #for r, rp in zip(answers, predictions):
    #    print(r, rp, r - rp)
    return np.sqrt(np.mean((predictions - answers) ** 2))