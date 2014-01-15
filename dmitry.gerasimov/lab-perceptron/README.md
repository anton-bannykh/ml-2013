Linear perceptron training home assignment

**common/bcwd.py**: BCWD data set related routines

**common/common.py**: random useful machine learning related functions

**lab-perceptron/perceptron.py**: generic perceptron training algorithm

**lab-perceptron/main.py**:

1. downloads the BCWD dataset
2. splits into training and test set (using the given fraction, 0.10 currently)
3. runs given number of perceptron training algorithm
4. calculates the error rate, precision and recall
5. averages error rate, precision, recall and f1 score over the given number of iteration and prints them out

Current error rate averaged over 100 runs is about 11 %.

You'll need `python3` and `numpy` to run the program.

To run, enter the `dmitry.gerasimov` directory and run `./run.sh lab-perceptron` (this script sets up python environment
to use files in the `common` dir shared among home assignments).
