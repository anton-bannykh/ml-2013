#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def bound(L, H, x):
    if x <= L:
        return L
    if x >= H:
        return H
    return x

def optimize_pair(t1, t2, C, f, kernel):
    a1, x1, y1 = t1
    a2, x2, y2 = t2

    # define clipping bounds
    if y1 != y2:
        L = max(0, a2 - a1)
        H = C + min(0, a2 - a1)
    else:
        L = max(0, a1 + a2 - C)
        H = min(C, a1 + a2)

    E1 = f(x1) - y1
    E2 = f(x2) - y2
    n = 2 * kernel(x1, x2) - kernel(x1, x1) - kernel(x2, x2)

    r2 = bound(L, H, a2 - y2 * (E1 - E2) / n)
    r1 = a1 + y1 * y2 * (a2 - r2)

    return r1, r2

def main():
    pass

if __name__ == "__main__":
    main()