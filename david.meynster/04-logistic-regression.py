from ml import logistic
from ml.data import *

def optimize_regularization(data):
    reg_best, err_best = 0, 10
    data_train, data_test = split(data)
    for d in range(-40, 10):
        reg_current = 2 ** d
        theta = logistic.train(data_train, l=reg_current)
        err_current = logistic.average_error(data_test, theta)
        if err_best > err_current:
            reg_best, err_best = reg_current, err_current
    return reg_best

def main():
    data = dataset_block(load_cancer_dataset(), withOnes=False)
    reg_const = optimize_regularization(data)
    data_train, data_test = split(data)
    theta = logistic.train(data_train, l=reg_const)
    err = logistic.average_error(data_test, theta)

    print("average error:%6.2f\n" % err)
    print("regularization constant used: %6.2f\n" % reg_const)

if __name__ == '__main__':
    main()