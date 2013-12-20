import random
import datetime
import sys

__author__ = 'jambo'
import numpy as np

MAX_ITER = 100

class DataSet:
    def __init__(self, file_path, answers_file_path):
        with open(file_path, 'r') as file:
            self.max_rating, self.n_users, self.n_items, self.n_train_rates, self.n_test_rates \
                = map(int, file.readline().split())
            self.ratings = np.zeros((self.n_users, self.n_items))
            self.ratings_as_list = []
            self.tests = []
            for _ in xrange(self.n_train_rates):
                u, i, r = map(int, file.readline().split())
                self.ratings[(u, i)] = r
                self.ratings_as_list.append((u, i, r))
            for _ in xrange(self.n_test_rates):
                self.tests.append((map(int, file.readline().split())))  # user - item
            with open(answers_file_path, 'r') as answers_file:
                self.answers = np.array(map(int, answers_file.readlines()))

class SVDModel:
    def __init__(self, dataset, num_of_factors, regularizations_constant, learning_rate):
        self.dataset = dataset
        self.average = self._average_rating()
        self.b_users = np.zeros(dataset.n_users)
        self.b_items = np.zeros(dataset.n_items)
        self.p = np.random.random((dataset.n_users, num_of_factors)) - 0.5
        self.q = np.random.random((dataset.n_items, num_of_factors)) - 0.5
        self.regularization_constant = regularizations_constant
        self.learning_rate = learning_rate
        self.validate_set_size = int(len(self.dataset.tests) * 0.2)
        self.size = len(self.dataset.tests)

    def predict(self, u, i):
        return self.average + self.b_users[u] + self.b_items[i] + np.inner(self.p[u], self.q[i])

    def fit_model(self):
        self._sgd()

    def rmse(self, cut=None):
        if cut is None:
            cut = self.size
        estimate = np.array([self.predict(u, i) for u, i in self.dataset.tests])[:cut]
        answers = self.dataset.answers[:cut]
        return float(np.sqrt(np.mean((estimate - answers) ** 2)))

    def _average_rating(self):
        return np.average(self.dataset.ratings[self.dataset.ratings > 0])

    def _error(self, u, i):
        return self.dataset.ratings[(u, i)] - self.predict(u, i)

    def validated_rmse(self):
        return self.rmse(cut=self.validate_set_size)

    def _sgd(self):
        gamma = self.learning_rate
        lam = self.regularization_constant
        previous_rmse = None
        for _ in xrange(MAX_ITER):
            random.shuffle(self.dataset.ratings_as_list)
            for u, i, r in self.dataset.ratings_as_list:
                error = self._error(u, i)
                new_b_u = self.b_users[u] + gamma * (error - lam * self.b_users[u])
                new_b_i = self.b_items[i] + gamma * (error - lam * self.b_items[i])
                new_p_u = self.p[u] + gamma * (error * self.q[i] - lam * self.p[u])
                new_q_i = self.q[i] + gamma * (error * self.p[u] - lam * self.q[i])
                self.b_users[u], self.b_items[i], self.p[u], self.q[i] = new_b_u, new_b_i, new_p_u, new_q_i
            new_rmse = self.validated_rmse()
            print "validate rmse: %0.5f" % new_rmse
            if previous_rmse is not None and previous_rmse - new_rmse < 5e-4:
                break

            previous_rmse = new_rmse


def grid_search():
    global results, learning_rate, factor_number, regularization_constant, model, time, rmse
    results = []
    for learning_rate in [0.005]:
        for factor_number in [0, 5, 10, 50, 100]:
            print "factor number = %d" % factor_number
            for regularization_constant in [0.05, 0.1, 0.5, 1, 5]:
                model = SVDModel(dataset, 50, regularization_constant, learning_rate)
                time = datetime.datetime.now()
                model.fit_model()
                print "seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds()
                rmse = model.rmse()
                print ("rmse for learning rate %0.4f and regularisation constant %0.4f: %0.5f"
                       % (learning_rate, regularization_constant, rmse))
                results.append((rmse, factor_number, learning_rate, regularization_constant))
    print "done"
    for rmse, factor_number, learning_rate, regularization_constant in sorted(results):
        print ("rmse for factor_number %d, learning rate %0.4f and regularisation constant %0.4f: %0.5f"
               % (factor_number, learning_rate, regularization_constant, rmse))


if __name__ == '__main__':

    # rmse for factor_number 0, learning rate 0.0050 and regularisation constant 0.0500: 0.86115
    # rmse for factor_number 5, learning rate 0.0050 and regularisation constant 0.0500: 0.86310
    # rmse for factor_number 50, learning rate 0.0050 and regularisation constant 0.0500: 0.86313
    # rmse for factor_number 100, learning rate 0.0050 and regularisation constant 0.0500: 0.86773

    if len(sys.argv) > 1:
        dataset_number = sys.argv[1]
    else:
        dataset_number = "1"
    dataset = DataSet("movielensfold%s.txt" % dataset_number, "movielensfold%sans.txt" % dataset_number)
    print "dataset loaded"
    model = SVDModel(dataset, 5, 0.05, 0.005)
    time = datetime.datetime.now()
    model.fit_model()
    print "seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds()
    print "rmse: %0.5f" % model.rmse()

    # grid_search()










