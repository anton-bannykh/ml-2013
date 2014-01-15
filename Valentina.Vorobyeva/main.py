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
    o = 0.
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
    #print(learn.perc)
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

    p = float(tp) / (tp + fp)
    r = float(tp) / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print ("Precision = " + str(p))
    print ("Recall = " + str(r))
    print ("F1-metric = " + str(f1))

    f.close()


if __name__ == '__main__':
    main()