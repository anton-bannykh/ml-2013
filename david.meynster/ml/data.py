from urllib.request import urlopen
import numpy as np
from os.path import exists

def load_cancer_dataset():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    local_path = "wdbc.data"
    try:
        if not exists(local_path):
            local_copy = open(local_path, 'wb')
            local_copy.write(urlopen(url))
            local_copy.close()
        data = open(local_path, "r")
    except Exception:
        data = urlopen(url)
    lines = data.readlines()
    n, d = len(lines), len(lines[0].split(',')) - 2
    x, y = np.empty((n, d)), np.empty(n)

    for i in range(n):
        line = lines[i].split(',')
        x[i] = np.array([float(f) for f in line[2:]])
        y[i] = 1 if line[1] == 'M' else -1

    return x, y

def split(data, fraction=0.2, shuffle=True):
    if shuffle:
        np.random.shuffle(data)
    point = int(len(data) * fraction)
    return data[point:], data[:point]

def split_xy(data):
    return data[:, :-1], data[:, -1]

def join_xy(data_x, data_y):
    return np.concatenate((data_x, data_y), axis=1)

def dataset_block(dataset, withOnes=True):
    x, y = dataset
    if withOnes:
        ones = np.empty((len(x), 1))
        ones.fill(1.0)
        x = join_xy(x, ones)
    return join_xy(x, y.reshape((len(y), 1)))
