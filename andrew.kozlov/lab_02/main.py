from lab_01.main import divide, percent, load_data
from lab_02.svm import optimize_c, calculate_error, fit_svm


__author__ = 'adkozlov'


def main():
    train_set, test_set = divide(load_data())

    c = optimize_c(train_set)
    w, b = fit_svm(train_set, c)

    e = calculate_error(test_set, w, b)
    print('C = %f\nerror = %6.2f' % (c, percent(e)))


if __name__ == "__main__":
    main()