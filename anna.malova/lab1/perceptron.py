from data import *
import numpy as np

class Perceptron:
    def __init__(self, vec, steps):
        self.theta = np.zeros(vec[0].data.size)
        for i in range(steps):
            for v in vec:
                if self.get_label(v) != v.label:
                    self.theta += v.label * v.data
        print("Training completed")


    def get_label(self, v):
        return np.sign(np.dot(self.theta.T, v.data))


def main(train_fraction, steps):
    train, test = load_data(train_fraction)
    classifier = Perceptron(train, steps)

    print_results(classifier, test)

if __name__ == "__main__":
    main(0.3, 1000)
