Logistic regression

You'll need `python3`, `numpy` and `scipy`.

You might need to install `gfortran` package on ubuntu.


To run, enter the `dmitry.gerasimov` directory and do `./run.sh lab-logistic` (this script sets up python environment to use
files in the `common` dir shared among home assignments).

**common/bcwd.py**: BCWD data set related routines

**common/common.py**: random useful machine learning related functions

**lab-svm/logistic.py**: logistic unit training

**lab-svm/main.py**:

1. downloads the BCWD dataset
2. splits into training, test and validation set (currently 0.8 : 0.1 : 0.1)
3. searches for best regularisation constant using validation data
4. calculates result on the test data
5. averages error rate, precision, recall and f1 score over the given number of iteration and prints them out


Current results:

* Error rate is about 5 %.