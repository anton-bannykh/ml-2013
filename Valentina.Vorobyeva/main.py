__author__ = 'valentinka'

import learn

perc = learn.perc
tp = 0
fp = 0
fn = 0
tn = 0


def test(x):
    global tp, tn, fp, fn
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
    if t == 1 and o == 1:
        tp += 1
    if t == 1 and o == -1:
        fn += 1
    if t == -1 and o == 1:
        fp += 1
    if t == -1 and o == -1:
        tn += 1


def main():
    learn.learn()
    print(learn.perc)
    f = open('wdbcl.data', 'rU')
    datas = []
    for line in f:
        data = line.split(',')
        del data[0]
        datas.append(data)

    for data in datas:
        test(data)

    print "tn", tn
    print "tp", tp
    print "fn", fn
    print "fp", fp

    if tp == 0 and fp == 0:
        precis = 0.0
    else:
        precis = float(tp) / (tp + fp)
    if tp == 0 and fn == 0:
        rec = 0.0
    else:
        rec = float(tp) / (tp + fn)
    print ("Precision = " + str(precis))
    print ("Recall = " + str(rec))

    f.close()


if __name__ == '__main__':
    main()