Linear perceptron training home assignment

**perceptron.py**: generic perceptron training algorithm

**bcwd.py**: BCWD data set related routines

**main.py**:

1. downloads the BCWD dataset
2. splits into training and test set (using the given fraction, 0.10 currently)
3. runs given number of perceptron training algorithm
4. calculates the error rate, precision and recall
5. averages error rate over the given number of iteration and prints it out

Current error rate averaged over 100 runs is about 11 %.