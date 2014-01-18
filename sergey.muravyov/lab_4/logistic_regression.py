import math
import numpy as np
from urllib.request import urlopen

def scale(x):
    x_min, x_max = np.array([1e10] * len(x[0])), np.zeros(len(x[0]))
    for xc in x:
        for i in range(len(xc)):
            x_min[i] = min(xc[i], x_min[i])
            x_max[i] = max(xc[i], x_max[i])
    for xc in x:
        for i in range(len(xc)):
            if (x_max[i] != x_min[i]):
                xc[i] = (xc[i] - x_min[i]) / (x_max[i] - x_min[i])
            else:
                xc[i] = 1
    return x;

def get_data():
    x, y = [], []
    file = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
    for line in file.readlines():
        input = line.decode('utf-8').strip().split(',')
        if input[1] == 'M':
            y.append(1.0)
        else:
            y.append(-1.0)
        num = [float(x) for x in input[2:]]
        num.insert(0, 1.0)
        num = np.array(num)
        x.append(num)
    file.close()
    x = scale(x)
    return shuffle(x, y) 

def shuffle(x, y):
    tmp = list(zip(x, y))
    np.random.shuffle(tmp)
    x1, y1 = [], []
    for i in range(len(tmp)):
        x1.append(tmp[i][0])
        y1.append(tmp[i][1])
    return x1, y1

def stop_Q(w1, w2, d):
    dif = w2 - w1
    norm = np.sqrt(np.inner(dif, dif))
    flag = 0
    if d < norm and d != 0:
        flag = 1
    else:
        d = norm
    if (norm < 0.1) and not (flag == 0):
         flag = 1
    return flag, d

def get_w(x, y, reg):
    wlen = len(x[0])
    w = np.zeros(wlen)
    dif_old = 0
    while (True):
        w_old = np.array(w)
        xc, yc = shuffle(x, y)
        xclen = len(xc)
        for i in range(xclen):            
            grad = np.zeros(wlen)
            for j in range(wlen):
                margin = yc[i] * np.inner(w, xc[i])
                if margin < 20:
                    grad[j] += (1 / (1 + math.exp(margin))) * yc[i] * xc[i][j]
                if j > 0:
                    grad[j] += w[j] * reg
            w += grad * 0.01
            
        flag, dif_old = stop_Q(w_old, w, dif_old)
        if flag == 1:
            break
    return w

def classification(x, y, pr):
    mc = 0
    for j in range(len(x)):
        if y[j] != pr[j]:
            mc += 1
    return mc / len(x)

def predict_logreg(x_out, w):
    return [-1.0 if 1 / (1 + math.exp(-np.inner(x, w))) < 0.5 else 1.0 for x in x_out]

def split_xy(x, y, i, p):
    return x[i * p: (i + 1) * p], x[:i * p] + x[(i + 1) * p:], y[i * p: (i + 1) * p], y[:i * p] + y[(i + 1) * p:]

def get_c(x, y, parts):
    part_size = int(len(x) / parts)
    best_C, best_err = 0, 10
    for d in range(0, 10):
        C = 3.0 ** -d
        print(C)
        cur_err = 0
        for i in range(parts):
            test_x, train_x, test_y, train_y = split_xy(x, y, i, part_size)
            w = get_w(train_x, train_y, C)
            pr = predict_logreg(test_x, w)
            cur_err += classification(test_x, test_y, pr)/parts
        if best_err > cur_err:
            best_err = cur_err
            best_C = C
    return best_C

def main():
    x, y = get_data()
    bound = int(len(x)*0.2)
    train_x, train_y, test_x, test_y = x[bound:], y[bound:], x[:bound], y[:bound]
    C = get_c(train_x, train_y, 10)
    w = get_w(train_x, train_y, C)
    pr = predict_logreg(train_x, w)
    err = classification(train_x, train_y, pr)
    
    print("regularization constant = %6.5f" % C)
    print('average error = %6.2f' % err)

if __name__ == '__main__':
    main()
