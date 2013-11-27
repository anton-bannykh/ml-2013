from common import *
from scipy.optimize import minimize

g = lambda z : 1. / (1 + numpy.exp(-z))

def mkLogisticRegressionClassifier(X, y, lam = 1):
    n, m  = X.shape
    def f(theta_):
        theta, theta0 = theta_[:m], theta_[m]
        res = y * (numpy.dot(X, theta) + theta0)
        return .5 * lam * sum(theta * theta) + sum(numpy.log(1. + numpy.exp(-y * (numpy.dot(theta, X.T) + theta0))))
    theta = minimize(f, numpy.zeros(m + 1)).x
    return lambda v : numpy.sign(numpy.dot(numpy.hstack((v, 1)).T, theta)) or 1

X, y = loadData()

bestLam, bestLamF1, bestRes = None, None, None

for lam in 2. ** numpy.arange(-5, 20):
    XStudy, yStudy, XTest, yTest = split(X, y, int(0.5 * y.size))
    classifier = mkLogisticRegressionClassifier(XStudy, yStudy, lam)
    res = checkClassifier(classifier, XTest, yTest)
    f1Here = f1(res)
    print ("Logistic regression with lambda=%f, F1=%f"%(lam, f1Here))
    if not bestLamF1 or f1Here > bestLamF1:
        bestLamF1, bestLam, bestRes = f1Here, lam, res

for (w, f) in [('accurancy', accurancy), ('precision', precision), ('recall', recall), ('F1 measure', f1)]:
    print ("Logistic regression with lambda=%f %s: %f"%(bestLam, w, f(bestRes)))
