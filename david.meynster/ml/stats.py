class Stats:
    tp = tn = fp = fn = 0

    def precision(self):
        return self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0

    def recall(self):
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0

    def error(self):
        return (self.fp + self.fn) / (self.tp + self.fp + self.fn + self.tn)

    def f_score(self, beta=1):
        p = self.precision()
        r = self.recall()
        return (1 + beta) * p * r / (beta * beta * p + r) if max(p, r) > 0 else 0