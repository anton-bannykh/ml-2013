import math


def norm2(r1, sub=0):
    ans = sum((r - sub) ** (r - sub) for _, r in r1)
    return math.sqrt(ans)


def mean(r1):
    ans = sum(r for _, r in r1)
    return ans / len(r1)


def cosine(r1, r2):
    ans = 0
    for item, rating in r1:
        if item in r2:
            ans += rating * r2[item]
    return ans / norm2(r1) / norm2(r2)


def pc(r1, r2):
    mean1, mean2 = mean(r1), mean(r2)
    ans = 0
    for item, rating in r1:
        if item in r2:
            ans += (rating - mean1) * (r2[item] - mean2)
    return ans / norm2(r1, sub=mean1) / norm2(r2, sub=mean2)