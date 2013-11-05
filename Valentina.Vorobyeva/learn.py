__author__ = 'valentinka'

import random

perc = []


def exac_learn(x):
    global perc
    eta = 0.05
    if x[0] == 'M':
        t = 1
    else:
        t = -1
    x[0] = 1
    for i in range(len(x)):
        x[i] = float(x[i])

    o = 0
    for i in range(len(x)):
        o += perc[i] * x[i]
    if o >= 0:
        o = 1
    else:
        o = -1
    for i in range(len(x)):
        perc[i] += eta * (t - o) * x[i]


def learn():
    f = open('wdbcl.data', 'rU')
    datas = []
    for line in f:
        data = line.split(',')
        del data[0]
        datas.append(data)

    for i in range(len(datas[0])):
        perc.append((random.randrange(0, 5)) / 500.0)

    for i in range(100):
        for data in datas:
            exac_learn(data)

    f.close()
