import math


def rmse(rs, realRs):
    return math.sqrt(mse(rs, realRs))


def mse(rs, realRs):
    ans = sum((r - realR) ** 2 for r, realR in zip(rs, realRs))
    return ans / len(rs)


def mae(rs, realRs):
    ans = sum(math.fabs(r - realR) for r, realR in zip(rs, realRs))
    return ans / len(rs)