__author__ = 'valentinka'

import math


class Point:
    def __init__(self, cs, l):
        self.cs = cs
        self.label = l


    def __repr__(self):
        ret = ''
        for c in self.cs:
            ret += '%s ' % c
        ret += '\'%s\'' % self.label
        return ret

    def metr(self, p, pres):
        dist = 0
        for i in xrange(len(self.cs)):
            for j in xrange(len(self.cs)):
                if abs(p.cs[i][j] - self.cs[i][j]) > pres:
                    dist += 1
        return dist
