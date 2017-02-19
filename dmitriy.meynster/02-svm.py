from ml import svm
from ml.data import *

def optimize_regularization(data):
    reg_best, f1_best = 0, 0
    data_train, data_test = split(data)
    for d in range(-10, 40):
        reg_current = 0.5 ** d
        theta = svm.train(data_train, c=reg_current)
        stats = svm.test(data_test, theta)
        f1_current = stats.f_score()
        if f1_best < f1_current:
            reg_best, f1_best = reg_current, f1_current
    return reg_best

def main():
    data = dataset_block(load_cancer_dataset(), withOnes=False)
    data_train, data_test = split(data)
    reg_const = optimize_regularization(data)
    theta = svm.train(data_train, c=reg_const)
    stats = svm.test(data_test, theta)

    print("precision:%6.2f\nrecall:%6.2f\nerror:%6.2f\nf1-score:%6.2f\n" %
          (stats.precision(), stats.recall(), stats.error(), stats.f_score()))
    print("regularization constant used: %6.2f\n" % reg_const)

if __name__ == '__main__':
    main()