__author__ = 'valentinka'

import sys
from data_read import read


def k_near_neigh(p, k, pres):
    global train_ps
    dist = {}
    for point in train_ps:
        dist[point] = p.metr(point, pres)
    d = sorted(dist.items(), key=lambda (k, v): v)
    ret = []
    for (key, value) in d:
        if len(ret) == k:
            break
        ret.append(key)
    return ret


def define_p(p, k, pres):
    neigh = k_near_neigh(p, k, pres)
    kand = {}
    for n in neigh:
        l = n.label
        if l in kand.keys():
            kand[l] += 1
        else:
            kand[l] = 1
    max = 0
    res = 0
    for key, value in kand.iteritems():
        if value > max:
            max = value
            res = key
    return res


def main(argv):
    global train_ps

    train_image_name = 'data/train-images.idx3-ubyte'
    train_label_name = 'data/train-labels.idx1-ubyte'
    test_image_name = 'data/t10k-images.idx3-ubyte'
    test_label_name = 'data/t10k-labels.idx1-ubyte'
    train_ps = read(train_image_name, train_label_name, 30000)
    test_ps = read(test_image_name, test_label_name, 5000)
    k = int(argv[1])
    pres = int(argv[2])
    right = 0
    wrong = 0
    for i, p in enumerate(test_ps):
        #print i
        l = define_p(p, k, pres)
        if p.label == l:
            right += 1
        else:
            wrong += 1
    print "k =", k, "pres =", pres
    res = "mistake =", float(wrong) * 100 / (wrong + right)
    print res


main(sys.argv)