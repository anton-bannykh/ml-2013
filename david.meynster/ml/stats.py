class Stats:
    tp = tn = fp = fn = 0

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def error(self):
        total = self.tp + self.fp + self.fn + self.tn
        return (self.fp + self.fn) / total

    def f_score(self, beta=1):
        p = self.precision()
        r = self.recall()
        return (1 + beta) * p * r / (beta * beta * p + r)