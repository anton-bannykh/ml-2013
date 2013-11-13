from common import *
from scipy.optimize import minimize

def mkSVMClassifier(X, y, C = 1):
    n, m  = X.shape
    def f(theta_):
        theta, theta0 = theta_[:m], theta_[m]
        res = y * (numpy.dot(X, theta) + theta0)
        return .5 * sum(theta * theta) + sum(1 - res[res < 1])
    theta = minimize(f, numpy.zeros(m + 1)).x
    return lambda v : numpy.sign(numpy.dot(numpy.hstack((v, 1)).T, theta)) or 1

X, y = loadData()

bestC, bestCF1, bestRes = None, None, None

for C in 2. ** numpy.arange(-5, 20):
    XStudy, yStudy, XTest, yTest = split(X, y, int(0.5 * y.size))
    classifier = mkSVMClassifier(XStudy, yStudy, C)
    res = checkClassifier(classifier, XTest, yTest)
    f1Here = f1(res)
    print ("SMV with C=%f, F1=%f"%(C, f1Here))
    if not bestCF1 or f1Here > bestCF1:
        bestCF1, bestC, bestRes = f1Here, C, res

for (w, f) in [('accurancy', accurancy), ('precision', precision), ('recall', recall), ('F1 measure', f1)]:
    print ("SVM with C=%f %s: %f"%(bestC, w, f(bestRes)))
