from ml import neural_network
from ml.data import *

def optimize_parameters(data):
    l_best, alpha_best, lsize_best = 2, 0.1, 100
    fscore_best, nn_best = 0.0, None
    alphas = map(lambda x: 0.33 ** x, range(3, 7))
    data_train, data_test = split(data)
    for layers in range(2, 6):
        for alpha in alphas:
            for lsize in [10, 50, 100]:
                nn = neural_network.train(data_train, layers, alpha, lsize)
                fscore = neural_network.test(data_test, nn).f_score()
                print(layers, alpha, lsize, ": ", fscore)
                if fscore > fscore_best:
                    fscore_best, nn_best = fscore, nn
                    l_best, alpha_best, lsize_best = layers, alpha, lsize
    return nn_best, (l_best, alpha_best, lsize_best)

def main():
    data = scale_features(dataset_block(load_cancer_dataset()))
    data_train, data_test = split(data)
    nn, params = optimize_parameters(data_train)
    stats = neural_network.test(data_test, nn)

    print("precision:%6.2f\nrecall:%6.2f\nerror:%6.2f\nf1-score:%6.2f\n" %
          (stats.precision(), stats.recall(), stats.error(), stats.f_score()))
    print("parameters used:\n  layers: %i\n  alpha: %i\n  lsize: %i\n" % params)

if __name__ == '__main__':
    main()