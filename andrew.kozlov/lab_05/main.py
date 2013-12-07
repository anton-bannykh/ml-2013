from lab_01.main import divide, load_data, percent
from lab_05.neural_network import calculate_error, optimize_size_lambda, thetas

__author__ = 'adkozlov'


def main():
    train_set, test_set = divide(load_data(negative=0))

    size, lambda_c = optimize_size_lambda(train_set, 2)
    theta1, theta2 = thetas(train_set, size, lambda_c, 2)

    print('hidden layer size = %d' % size)
    print('lambda = %f' % lambda_c)
    print('error = %6.2f' % percent(calculate_error(test_set, theta1, theta2)))


if __name__ == "__main__":
    main()