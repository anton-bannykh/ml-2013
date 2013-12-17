import random
import datetime

__author__ = 'jambo'
import numpy as np

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
        self.b_users = np.ones(dataset.n_users)
        self.b_items = np.ones(dataset.n_items)
        self.p = np.ones((dataset.n_users, num_of_factors))
        self.q = np.ones((dataset.n_items, num_of_factors))
        self.regularization_constant = regularizations_constant
        self.learning_rate = learning_rate

    def predict(self, u, i):
        return self.average + self.b_users[u] + self.b_items[i] + np.inner(self.p[u], self.q[i])

    def fit_model(self):
        self._sgd()

    def rmse(self):
        estimate = np.array([self.predict(u, i) for u, i in self.dataset.tests])
        answers = self.dataset.answers
        return float(np.sqrt(np.mean((estimate - answers) ** 2)))

    def _average_rating(self):
        return np.average(self.dataset.ratings[self.dataset.ratings > 0])

    def _error(self, u, i):
        return self.dataset.ratings[(u, i)] - self.predict(u, i)

    def _sgd(self):
        gamma = self.learning_rate
        lam = self.regularization_constant
        for _ in xrange(30):
            print _
            random.shuffle(self.dataset.ratings_as_list)
            for u, i, r in self.dataset.ratings_as_list:
                error = self._error(u, i)
                new_b_u = self.b_users[u] + gamma * (error - lam * self.b_users[u])
                new_b_i = self.b_items[i] + gamma * (error - lam * self.b_items[i])
                new_p_u = self.p[u] + gamma * (error * self.q[i] - lam * self.p[u])
                new_q_i = self.q[i] + gamma * (error * self.p[u] - lam * self.q[i])
                self.b_users[u], self.b_items[i], self.p[u], self.q[i] = new_b_u, new_b_i, new_p_u, new_q_i
            print self.rmse()


if __name__ == '__main__':
    dataset = DataSet("movielensfold1.txt", "movielensfold1ans.txt")
    print "dataset loaded"

    results = []
    for factor_number in [0, 5, 10, 50, 100]:
        print "factor number = %d" % factor_number
        for learning_rate in [0.001, 0.005, 0.01, 0.05, 0.1]:
            for regularization_constant in [10 ** x for x in xrange(-5, 3)]:
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









