from cancer_common.reader import *
from cancer_svm.svm import *

def get_best_constant(x, y, parts):
    part_size = int(len(x) / parts)
    minEout = 1
    best_const = 1

    for deg in range(-5, 20):
        const = (2.0) ** (deg)
        print(deg)
        Eout = 0

        for i in range(parts):
            check_set_x = x[i * part_size: (i + 1) * part_size]
            check_set_y = y[i * part_size: (i + 1) * part_size]

            training_set_x = x[:i * part_size] + x[(i + 1) * part_size:]
            training_set_y = y[:i * part_size] + y[(i + 1) * part_size:]

            w, b = get_w(training_set_x, training_set_y, const)


            prediction = predict(check_set_x, w, b)

            misclassified = 0
            for j in range(len(check_set_x)):
                if check_set_y[j] != prediction[j]:
                    misclassified += 1
            Eout += misclassified / len(check_set_x) / parts

        if Eout < minEout:
            minEout = Eout
            best_const = const

    return best_const

TEST_PERCENT = 0.2
PARTS = 10

x, y = get_data()
test_size = int(len(x) * TEST_PERCENT)

check_set_x = x[:test_size]
check_set_y = y[:test_size]

training_set_x = x[test_size:]
training_set_y = y[test_size:]

const = get_best_constant(training_set_x, training_set_y, PARTS)

w, b = get_w(training_set_x, training_set_y, const)
prediction = predict(training_set_x, w, b)

misclassified = 0
for j in range(len(training_set_x)):
    if training_set_y[j] != prediction[j]:
        misclassified += 1
Ein = misclassified / len(training_set_x)

prediction = predict(check_set_x, w, b)

# 'tp' — true positive
# 'tn' — true negative
# 'fp' — false positive
# 'fn' — false negative
stats = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

for j in range(len(check_set_x)):
    if check_set_y[j] == -1:
        key = 'tn' if prediction[j] == -1 else 'fp'
        stats[key] += 1
    if check_set_y[j] == 1:
        key = 'tp' if prediction[j] == 1 else 'fn'
        stats[key] += 1

precision = stats['tp'] / (stats['tp'] + stats['fp'])
recall = stats['tp'] / (stats['tp'] + stats['fn'])
F1 = 2 * precision * recall / (precision + recall)
Eout = (stats['fp'] + stats['fn']) / len(x[:test_size])

print("regularization constant = %6.5f" % const)
print('in sample error         = %6.2f' % (100 * Ein))
print('out of sample error     = %6.2f' % (100 * Eout))
print('precision               = %6.2f' % (100 * precision))
print('recall                  = %6.2f' % (100 * recall))
print('F1                      = %6.2f' % (100 * F1))
