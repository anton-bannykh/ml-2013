__author__ = 'Den'

import numpy as nu


data = []

def dot(x, teta):
    return nu.dot(x[:-1], teta)

def readData():
    res = []
    file = open("wdbc.data")
    for line in file:
        arr = line.split(',');
        res.append(arr[2:])
        res[-1].append(1 if arr[1] == 'M' else -1)
        res[-1] = [float(res[-1][i]) for i in range(len(res[-1]))]
    return res

def learn():
    teta = [0 for _ in range(0,30)]
    global data
    data = readData()
    for x in data[:len(data) // 5]:
        if dot(x, teta) * x[-1] <= 0:
            teta = [teta[i] + x[i]*x[-1] for i in range(len(teta))]
    return teta

def check(teta):
    wrong = 0
    global data
    for x in data[len(data) // 5:]:
        if dot(x, teta) * x[-1] < 0:
            wrong += 1
    return wrong / (len(data) - len(data) // 5)

print(check(learn()))