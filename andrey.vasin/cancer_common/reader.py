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
            y.append(1)
        else:
            y.append(-1)

        parameters = [float(x) for x in input[2:]]
        parameters.insert(0, 1.0)
        parameters = numpy.array(parameters)
        x.append(parameters)
    file.close()
    return shuffle(x, y)
