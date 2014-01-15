from numpy import log, exp
from numpy.linalg import norm
import scipy.optimize as opt
from data import *

class logistic_regression:
    def __init__(self, vec, λ):
        x = np.array([v.data for v in vec])
        y = np.array([v.label for v in vec])
        def loss(theta_):
            n = theta_.shape[0]
            return 0.5 * λ * norm(theta_)**2 + sum(log(1 + exp(-y*(theta_[:-1].dot(x.T) + theta_[n - 1]))))

        self.theta = opt.minimize(loss, np.zeros(vec[0].data.size + 1)).x

        print("Training completed with λ = " + str(λ))

    def get_label(self, v):
        n = self.theta.shape[0]
        return np.sign(np.dot(self.theta[:-1].T, v.data) + self.theta[n - 1])

def main(train_fraction):
    train, test = load_data(train_fraction)

    best_λ = None
    best_score = None
    for λ in 2. ** np.arange(-15, 15):
    #for C in range(8, 9):
        cv_train, cv_test = split_data(train, 0.5)
        classifier = logistic_regression(cv_train, λ)
        score = get_metrics(classifier, cv_test)['f1']
        if best_score == None or score > best_score:
            best_score = score
            print("Now best score is " + str(best_score))
            best_λ = λ

    classifier = logistic_regression(train, best_λ)
    print("Best λ = " + str(best_λ))
    print_results(classifier, test)

if __name__ == "__main__":
    main(0.7)