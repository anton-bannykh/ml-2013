from ml import svm_smo
from ml.kernel import *
from ml.data import *

def main():
    data = dataset_block(load_cancer_dataset(), withOnes=False)
    data_train, data_test = split(data)
    print(len(data_train))
    reg_const = 200.0

    alpha_g, b_g = svm_smo.train(data_train, reg_const, kernel=gaussian)
    stats_g = svm_smo.test(data_train, data_test, (alpha_g, b_g), kernel=gaussian)
    print("SMO with gaussian kernel:")
    print("precision:%6.2f\nrecall:%6.2f\nerror:%6.2f\nf1-score:%6.2f\n" %
          (stats_g.precision(), stats_g.recall(), stats_g.error(), stats_g.f_score()))

    alpha_p, b_p = svm_smo.train(data_train, reg_const, kernel=polynomial)
    stats_p = svm_smo.test(data_train, data_test, (alpha_p, b_p), kernel=polynomial)
    print("SMO with polynomial kernel:")
    print("precision:%6.2f\nrecall:%6.2f\nerror:%6.2f\nf1-score:%6.2f\n" %
          (stats_p.precision(), stats_p.recall(), stats_p.error(), stats_p.f_score()))

if __name__ == '__main__':
    main()