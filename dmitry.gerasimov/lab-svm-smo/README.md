Support vector machine home assignment

You'll need `python3`, `numpy` and [`cvxopt`](http://cvxopt.org/).

You might need to install lapack (`liblapack3-dev` in ubuntu) and atlas (`libatlas-dev` in ubuntu) to compile `cvxopt`.

To run, enter the `dmitry.gerasimov` directory and do `./run.sh lab-svm-smo` (this script sets up python environment to use
files in the `common` dir shared among home assignments).

**common/bcwd.py**: BCWD data set related routines

**common/common.py**: random useful machine learning related functions

**lab-svm/svm_smo.py**: SVM training

**lab-svm/kernels.py**: kernels for SVM

**lab-svm/main.py**:

1. downloads the BCWD dataset
2. splits into training, test and validation set (currently 0.8 : 0.1 : 0.1)
3. searches for best regularisation constant using validation data
4. calculates result on the test data
5. averages error rate, precision, recall and f1 score over the given number of iteration and prints them out


Current results:

TODO
