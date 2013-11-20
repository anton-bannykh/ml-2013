import numpy
from cancer.tools import precision, recall, error_rate
from tools import training_and_test_sets


def get_answers_and_actual(learning_set, omega):
    actual = [test.answer for test in learning_set]
    answers = [numpy.sign(omega.dot(test.features.T)) for test in learning_set]
    return answers, actual


def learn_perceptron(learning_set):
    """
    @type learning_set: list of DataElement
    """
    omega = numpy.zeros(learning_set[0].features.shape)
    best_omega = omega.copy()
    actual = [test.answer for test in learning_set]
    answers = [numpy.sign(omega.dot(test.features.T)) for test in learning_set]
    best_error_rate = error_rate(answers, actual)
    for x in xrange(100):
        updates = 0
        for test in learning_set:
            if numpy.sign(omega.dot(test.features.T)) != test.answer:
                updates += 1
                omega += test.answer * test.features

        answers, actual = get_answers_and_actual(learning_set, omega)
        new_error = error_rate(answers, actual)

        if new_error < best_error_rate:
            best_error_rate = new_error
            best_omega = omega.copy()

        if updates == 0:
            break

    return best_omega


def main():
    training_set, test_set = training_and_test_sets()
    omega = learn_perceptron(training_set)
    real_answers = [test.answer for test in test_set]
    classification_answers = [numpy.sign(omega.dot(test.features.T)) for test in test_set]
    print "error_rate: %f" % error_rate(classification_answers, real_answers)
    print "precision: %f" % precision(classification_answers, real_answers)
    print "recall: %f" % recall(classification_answers, real_answers)


if __name__ == '__main__':
    main()