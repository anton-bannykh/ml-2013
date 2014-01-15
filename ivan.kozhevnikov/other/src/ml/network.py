from math import exp
import random


def activation(x):
    if x < -30:
        return 0
    if x > 30:
        return 1
    return 1 / (1 + exp(-x))


def d_activation(x):
    res = activation(x)
    return res * (1 - res)


def random_row(length, sum):
    row = []
    for j in xrange(length):
        row.append(random.uniform(-2. / sum, 2. / sum))
    return row


class NeuronNetwork:
    def __init__(self, layers):
        layers_matrix = []
        q = sum(layers)
        biases = [[0] * layers[0]]
        for layer in xrange(len(layers) - 1):
            matrix = []
            for i in xrange(layers[layer]):
                matrix.append(random_row(layers[layer + 1], q))
            layers_matrix.append(matrix)
            biases.append(random_row(layers[layer + 1], q))

        self.layers = layers
        self.layers_matrix = layers_matrix
        self.biases = biases

    def forward(self, input):
        inputs = [input]
        zs = [input]
        for layer in range(1, len(self.layers)):
            next_input = [0] * self.layers[layer]
            weights = self.layers_matrix[layer - 1]
            for i in xrange(0, self.layers[layer - 1]):
                for j in xrange(0, self.layers[layer]):
                    next_input[j] += zs[layer - 1][i] * weights[i][j]

            for j in xrange(0, self.layers[layer]):
                next_input[j] += self.biases[layer][j]
            zs.append([activation(x) for x in next_input])
            inputs.append(next_input)

        return inputs, zs

    def output(self, input):
        inputs, zs = self.forward(input)
        return zs[len(self.layers) - 1]

    def train(self, xs, ys, alpha):
        inputs, zs = self.forward(xs)
        deltas = [(ys[i] - zs[len(self.layers) - 1][i]) for i in xrange(len(ys))]

        for layer in range(len(self.layers) - 1, 0, -1):
            weights = self.layers_matrix[layer - 1]

            new_deltas = [0] * self.layers[layer - 1]
            for i in xrange(0, self.layers[layer - 1]):
                for j in xrange(0, self.layers[layer]):
                    new_deltas[i] += weights[i][j] * deltas[j]
                new_deltas[i] *= d_activation(inputs[layer - 1][i])

            for i in xrange(0, self.layers[layer - 1]):
                for j in xrange(0, self.layers[layer]):
                    weights[i][j] += alpha * deltas[j] * zs[layer - 1][i]

            for j in xrange(0, self.layers[layer]):
                self.biases[layer][j] += alpha * deltas[j]

            deltas = new_deltas


class NeuronClassifier:
    def __init__(self, c, vectors, old_classes):
        n = len(vectors[0])
        vertexes = n * int(c * 4)
        classes = map(lambda x: (x + 1) / 2, old_classes)
        network = NeuronNetwork([n, vertexes, 1])
        for i in range(5):
            for x, y in zip(vectors, classes):
                network.train(x, [y], 1)
        self.network = network

    def classify(self, x):
        res = self.network.output(x)[0]
        if res > 0.5:
            return 1
        else:
            return -1