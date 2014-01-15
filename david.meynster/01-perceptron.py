from ml import perceptron
from ml.data import *

def main():
    data = dataset_block(load_cancer_dataset())
    data_train, data_test = split(data)
    theta = perceptron.train(data_train)
    stats = perceptron.test(data_test, theta)

    print("precision:%6.2f\nrecall:%6.2f\nerror:%6.2f\nf1-score:%6.2f\n" %
          (stats.precision(), stats.recall(), stats.error(), stats.f_score()))

if __name__ == '__main__':
    main()