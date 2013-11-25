import regularization


def read_input(file):
    input_data = open(file)
    vectors = []
    classes = []
    for line in input_data:
        parts = line.split(",")
        vector = map(lambda x: float(x), parts[2:])
        clazz = -1
        if parts[1] == 'B':
            clazz = 1
        vectors.append(vector)
        classes.append(clazz)
    return vectors, classes


def calc_score(test, classifier):
    result = []
    for vector, clazz in test:
        result.append((clazz, classifier.classify(vector)))
    tp = 0.
    fp = 0.
    tn = 0.
    fn = 0.
    for x, y in result:
        if x == 1 and y == 1:
            tp += 1
        if x == 1 and y == -1:
            fn += 1
        if x == -1 and y == 1:
            fp += 1
        if x == -1 and y == -1:
            tn += 1
    error = (fp + fn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (fp + tp)
    return error, recall, precision


def f1_metric(error, recall, precision):
    return 2 * recall * precision / (error + precision)


def print_result(error, recall, precision):
    print("Error: {0:.4f}".format(error))
    print("Recall: {0:.4f}".format(recall))
    print("Precision: {0:.4f}".format(precision))


def run(method_builder, file):
    input_data = zip(*read_input(file))
    c = regularization.find_regularization_const(input_data, method_builder)

    result = regularization.build_classifier_get_result(input_data, c, method_builder)
    print("The best result is when C is: " + str(c))
    print_result(*result)