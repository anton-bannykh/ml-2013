#ifndef KERNELS_H
#define KERNELS_H

#include <cmath>
#include <vector>
#include <cassert>

typedef std::vector<double> array;

class LinearKernel {
public:
    double compute(const array& x, const array& y) const {
        assert(x.size() == y.size());
        double result = 0;

        for (size_t i = 0; i < x.size(); ++i) {
            result += x[i] * y[i];
        }

        return result;
    }
};

class PolynomialKernel {
private:
    int degree;
public:
    PolynomialKernel(int degree) : degree(degree) { }
    double compute(const array& x, const array& y) const {
        assert(x.size() == y.size());

        double dot_product = 0;

        for (size_t i = 0; i < x.size(); ++i) {
            dot_product += x[i] * y[i];
        }

        return pow(1 + dot_product, degree);
    }
};

class GaussianKernel {
private:
    double gamma;
public:
    GaussianKernel(double gamma) : gamma(gamma) { }
    double compute(const array& x, const array& y) const {
        assert(x.size() == y.size());

        double squared_norm = 0.0;

        for (size_t i = 0; i < x.size(); ++i) {
            double diff = x[i] - y[i];
            squared_norm += diff * diff;
        }

        return exp(-gamma * squared_norm);
    }
};

#endif // KERNELS_H
