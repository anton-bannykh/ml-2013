import numpy
from urllib.request import urlopen

URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

def shuffle(x, y):
    tmp = list(zip(x, y))
    numpy.random.shuffle(tmp)
    x_new = []
    y_new = []
    for i in range(len(tmp)):
        x_new.append(tmp[i][0])
        y_new.append(tmp[i][1])
    return x_new, y_new

def get_data():
    x, y = [], []
    file = urlopen(URL)
    for line in file.readlines():
        input = line.decode('utf-8').strip().split(',')

        if input[1] == 'M':
            y.append(1.0)
        else:
            y.append(-1.0)

        parameters = [float(x) for x in input[2:]]
        parameters.insert(0, 1.0)
        parameters = numpy.array(parameters)
        x.append(parameters)
    file.close()
    return shuffle(x, y)

def get_scaled_data():
    x, y = [], []
    file = urlopen(URL)
    for line in file.readlines():
        input = line.decode('utf-8').strip().split(',')

        if input[1] == 'M':
            y.append(1.0)
        else:
            y.append(-1.0)

        parameters = [float(x) for x in input[2:]]
        parameters.insert(0, 1.0)
        parameters = numpy.array(parameters)
        x.append(parameters)
    file.close()
    x_min, x_max = numpy.array([1e10] * len(x[0])), numpy.zeros(len(x[0]))
    for x_cur in x:
        for i in range(len(x_cur)):
            if x_min[i] > x_cur[i]:
                x_min[i] = x_cur[i]
            if x_max[i] < x_cur[i]:
                x_max[i] = x_cur[i]
    for x_cur in x:
        for i in range(len(x_cur)):
            if (x_max[i] != x_min[i]):
                x_cur[i] = (x_cur[i] - x_min[i]) / (x_max[i] - x_min[i])
            else:
                x_cur[i] = 1
    return shuffle(x, y)
