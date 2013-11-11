from cancer_common.reader import *
from cancer_perceptron.perceptron import *
import numpy

TEST_PERCENT = 0.1

x, y = get_data()
test_size = int(len(x) * TEST_PERCENT)

w, Ein = train_perceptron(x[test_size:], y[test_size:], 1000)

# 'tp' — true positive
# 'tn' — true negative
# 'fp' — false positive
# 'fn' — false negative
stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
for xCur, yCur in zip(x[:test_size], y[:test_size]):
    yc = classify(w, xCur)
    if yCur == -1:
        key = 'tn' if yc == -1 else 'fp'
        stats[key] += 1
    if yCur == 1:
        key = 'tp' if yc == 1 else 'fn'
        stats[key] += 1

precision = stats['tp'] / (stats['tp'] + stats['fp'])
recall = stats['tp'] / (stats['tp'] + stats['fn'])
F1 = 2 * precision * recall / (precision + recall)
Eout = (stats['fp'] + stats['fn']) / len(x[:test_size])

print('in sample error     = %6.2f' % (100 * Ein))
print('out of sample error = %6.2f' % (100 * Eout))
print('precision           = %6.2f' % (100 * precision))
print('recall              = %6.2f' % (100 * recall))
print('F1                  = %6.2f' % (100 * F1))