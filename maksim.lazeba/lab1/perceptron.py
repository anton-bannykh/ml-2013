from functools import reduce

__author__ = 'max'


class Perceptron:
    weights = []

    def __init__(self, weights):
        self.weights = weights

    def get_weights(self):
        return list(self.weights)

    def calc_output(self, xs):
        return self.calc_with_x0([1] + xs)

    def calc_with_x0(self, xs):
        if len(xs) != len(self.weights):
            raise ValueError
        return reduce(lambda res, x: res + x, map(lambda x, y: x * y, xs, self.weights), 0)

    def train(self, eta, train_set, max_iterations):
        changed = True
        while changed:
            if max_iterations == 0:
                break
            max_iterations -= 1
            changed = False
            for _, t, xs in train_set:
                xs = [1] + xs
                o = self.calc_with_x0(xs)
                o = 1 if o > 0 else -1
                if o * t <= 0:
                    changed = True
                    for i in range(len(xs)):
                        self.weights[i] += eta * (t - o) * xs[i]


def get_untraining_perceptron(entry_num):
    import random

    random.seed(1234)

    weights = [random.gauss(0, 0.01) for i in range(entry_num + 1)]
    return Perceptron(weights)