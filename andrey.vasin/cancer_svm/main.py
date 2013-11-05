from cancer_common.reader import *
from sklearn import svm

def get_best_constant(x, y, parts):
    part_size = int(len(x) / parts)
    minEout = 1
    best_const = 1

    for deg in range(30):
        const = (1.0 / 3.0) ** (deg)
        Eout = 0

        for i in range(parts):
            check_set_x = x[i * part_size: (i + 1) * part_size]
            check_set_y = y[i * part_size: (i + 1) * part_size]

            training_set_x = x[:i * part_size] + x[(i + 1) * part_size:]
            training_set_y = y[:i * part_size] + y[(i + 1) * part_size:]

            svmReg = svm.LinearSVC(C=const)
            svmReg.fit(training_set_x, training_set_y)


            prediction = svmReg.predict(check_set_x)

            misclassified = 0
            for j in range(len(check_set_x)):
                if check_set_y[j] != prediction[j]:
                    misclassified += 1
            Eout += misclassified / len(check_set_x) / parts

        if Eout < minEout:
            minEout = Eout
            best_const = const

    return best_const

TEST_PERCENT = 0.1
PARTS = 10

x, y = get_data()
test_size = int(len(x) * TEST_PERCENT)

check_set_x = x[:test_size]
check_set_y = y[:test_size]

training_set_x = x[:test_size]
training_set_y = y[:test_size]

const = get_best_constant(training_set_x, training_set_y, PARTS)

svmReg = svm.LinearSVC(C=const)
svmReg.fit(training_set_x, training_set_y)

prediction = svmReg.predict(check_set_x)

misclassified = 0
for j in range(len(check_set_x)):
    if check_set_y[j] != prediction[j]:
        misclassified += 1

Eout = misclassified / len(check_set_x)

print("regularization constant: %6.5f" % const)
print("out of sample error: %6.2f%%" % (100 * Eout))
