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

def load_movielens_dataset(number=0):
    train = open("itmo-recsys-data/movielensfold" + str(number) + ".txt", 'r')
    test = open("itmo-recsys-data/movielensfold" + str(number) + "ans.txt", 'r')
    _, nu, ni, ntrain, ntest = [int(x) for x in train.readline().split()]
    train_data, test_data = np.empty((ntrain, 3), dtype=int), np.empty((ntest, 3), dtype=int)
    for i in range(ntrain):
        train_data[i] = [int(x) for x in train.readline().split()]
    for i in range(ntest):
        test_data[i][:2] = [int(x) for x in train.readline().split()]
    # ans-file -- only ratings, add also u and i to test_data!!!
    for i in range(ntest):
        test_data[i][2] = int(test.readline())
    return train_data, test_data, (nu, ni)

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
