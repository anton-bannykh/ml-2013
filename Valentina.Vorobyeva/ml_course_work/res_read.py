__author__ = 'valentinka'

import re

def res_read(k):
    name = 'res_lk/results_%s' % k
    f = open(name, 'r')
    mist = []
    for line in f:
        res = re.findall('\d+\.\d*', line)
        if len(res) > 0:
            mist.append(float(res[0]))
        else:
            break
    f.close()
    print 'y%s = %s' % (k, mist)
    print 'ys.append(y%s)' % k

for k in range(1, 11):
     res_read(k)
