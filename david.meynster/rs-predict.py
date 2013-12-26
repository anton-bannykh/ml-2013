import numpy as np
import datetime
from ml.data import load_movielens_dataset, split
from ml.rs import svd_sgd
datasets_total = 5

def optimize_parameters(train, validate, lengths):
    rmse_best, params_best, pred_best = 100, ((0, 0), (0, 0)), lambda u, i: 0
    for lam1 in 0.1 ** np.arange(0.0, 3.0):
        for lam2 in 0.1 ** np.arange(0.0, 3.0):
            for gamma1 in 0.33 ** np.arange(5.0, 8.0):
                for gamma2 in 0.33 ** np.arange(5.0, 8.0):
                    #print(lam1, lam2, gamma1, gamma2)
                    predictor = svd_sgd.learn(train, lengths, (lam1, lam2), (gamma1, gamma2))
                    rmse = svd_sgd.rmse(validate, predictor)
                    #print(rmse)
                    if rmse < rmse_best:
                        rmse_best = rmse
                        params_best = ((lam1, lam2), (gamma1, gamma2))
                        pred_best = predictor
    return params_best, pred_best

def main():
    for number in range(1, 1 + datasets_total):
        #time = datetime.datetime.now()
        train, test, lengths = load_movielens_dataset(number)
        learn, validate = split(train, shuffle=False)
        (lam, gamma), predictor = optimize_parameters(learn, validate, lengths)
        print("Dataset #%i: RMSE = %6.4f" % (number, svd_sgd.rmse(test, predictor)))
        print("Parameters: λ1 = %6.4f, λ2 = %6.4f, ɣ1 = %6.4f, ɣ2 = %6.4f" %
              (lam[0], lam[1], gamma[0], gamma[1]))
        #print("seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds())

if __name__ == '__main__':
    main()