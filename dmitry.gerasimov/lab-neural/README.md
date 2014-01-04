Feedforward neural network

You'll need `python3`, `numpy` and `scipy`.

To run, enter the `dmitry.gerasimov` directory and do `./run.sh lab-neural` (this script sets up python environment to use
files in the `common` dir shared among home assignments).

**common/bcwd.py**: BCWD data set related routines

**common/common.py**: random useful machine learning related functions

**lab-svm/neural_network.py**: logistic unit training

**lab-svm/main.py**:

1. downloads the BCWD dataset
2. splits into training, test and validation set (currently 0.8 : 0.1 : 0.1)
3. chooses hidden layer size using validation data
4. calculates result on the test data
5. averages error rate, precision, recall and f1 score over the given number of iteration and prints them out


Current results:

* Average error rate (5 runs) is 6.07142857142857%
* Average precision (5 runs) is 90.15810276679842%
* Average recall (5 runs) is 94.28362573099415%
* Average f1 score (5 runs) is 0.915156732410737