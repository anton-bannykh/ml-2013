import numpy as np
from numpy.linalg import norm
import scipy.optimize as opt
from data import *

class svm:
    def __init__(self, vec, C):
        def loss(theta_):
            n = theta_.shape[0]
            a = np.array([v.label * (theta_[:-1].dot(v.data) + theta_[n - 1]) for v in vec])
            return 0.5 * norm(theta_)**2 + C * sum(1 - a[a <= 1])

        self.theta = opt.minimize(loss, np.zeros(vec[0].data.size + 1)).x

        print("Training completed with C = " + str(C))

    def get_label(self, v):
        n = self.theta.shape[0]
        return np.sign(np.dot(self.theta[:-1].T, v.data) + self.theta[n - 1])

def main(train_fraction):
    train, test = load_data(train_fraction)

    best_C = None
    best_score = None
    for C in 2. ** np.arange(-15, 15):
    #for C in range(8, 9):
        cv_train, cv_test = split_data(train, 0.5)
        classifier = svm(cv_train, C)
        score = get_metrics(classifier, cv_test)['f1']
        if best_score == None or score > best_score:
            best_score = score
            print("Now best score is " + str(best_score))
            best_C = C

    classifier = svm(train, best_C)
    print("Best C = " + str(best_C))
    print_results(classifier, test)

if __name__ == "__main__":
    main(0.7)