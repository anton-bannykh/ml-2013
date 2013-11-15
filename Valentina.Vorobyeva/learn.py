__author__ = 'valentinka'

import random

perc = []


def exac_learn((y, x)):
    global perc
    eta = 0.05
    o = 0.
    for i in range(len(x)):
        o += perc[i] * x[i]
    if o >= 0:
        o = 1.
    else:
        o = -1.
    if y - o == 0:
        return False
    else:
        for i in range(len(x)):
            perc[i] += eta * (y - o) * x[i]
        return True


def init(length):
    for i in range(length):
        perc.append((random.randrange(0, 5)) / 500.0)


def learn():
    f = open('wdbcl.data', 'rU')
    datas = []
    for line in f:
        data = line.split(',')
        if data[1] == 'M':
            y = 1.
        else:
            y = -1.
        data = (y, [1.] + [float(x) for x in data[2:]])
        datas.append(data)
    random.shuffle(datas)
    f.close()

    init(len(datas[0][1]))
    for i in range(1000):
        changed = False
        for data in datas:
            if exac_learn(data):
                changed = True
        if not changed:
            break
