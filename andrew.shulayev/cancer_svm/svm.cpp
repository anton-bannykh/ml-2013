#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstdio>

#include "kernels.h"

inline int random(int max) {
  return rand() % max;
}

inline double compute_unbiased_svm_value(const std::vector<array>& k, const array& ys, const array& a, int j) {
  double result = 0;

  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * ys[i] * k[i][j];
  }

  return result;
}

array operator*(double scale, const array& v) {
  array result(v.size());

  for (size_t i = 0; i < v.size(); ++i) {
    result[i] = scale * v[i];
  }

  return result;
}

void operator+=(array& left, const array& right) {
  assert(left.size() == right.size());
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] += right[i];
  }
}

inline double clip(double L, double R, double x) {
  if (x <= L)
    return L;
  if (x >= R)
    return R;
  return x;
}

template <typename Kernel>
array sequential_minimal_optimization
(
  const std::vector<array>& xs,
  const array& ys,
  const Kernel& kernel,
  double& b,
  double C,
  int max_passes = 10,
  double tolerance = 1e-5
) {
  assert(xs.size() == ys.size());
  const int count = xs.size(); // count of training set size

  array a(count, 0.0);
  b = 0.0;

  // k[i][j] is value of kernel(xs[i], xs[j])
  std::vector<array> k(count);
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < count; ++j) {
      double result = kernel.compute(xs[i], xs[j]);
      k[i].push_back(result);
    }
  }

  int passes = 0;
  while (passes < max_passes) {
    bool changed = false;

    for (int i = 0; i < count; ++i) {
      // j is a random index that isn't equal to i
      int j = random(count - 1);
      if (j == i) {
        j = count - 1;
      }

      double y1 = ys[i];
      double a1 = a[i];
      double E1 = compute_unbiased_svm_value(k, ys, a, i) + b - y1;
      
      bool kkt_violated = (y1 * E1 < -tolerance && a1 < C) || (y1 * E1 > tolerance && a1 > 0);

      if (!kkt_violated)
        continue;
      
      double L, H;
      double y2 = ys[j];
      double a2 = a[j];
      double E2 = compute_unbiased_svm_value(k, ys, a, j) + b - y2;
      if (y1 != y2) {
        L = std::max(0.0, a2 - a1);
        H = std::min(C, C + a2 - a1);
      } else {
        L = std::max(0.0, a1 + a2 - C);
        H = std::min(C, a1 + a2);
      }

      if (fabs(L - H) < tolerance)
        continue;

      double n = 2 * k[i][j] - k[i][i] - k[j][j];
      if (n >= 0)
        continue;

      double r2 = clip(L, H, a2 - y2 * (E1 - E2) / n);
      if (fabs(a2 - r2) < tolerance)
        continue;

      double r1 = a1 + y1 * y2 * (a2 - r2);

      double b1 = b - E1 - y1 * (r1 - a1) * k[i][i] - y2 * (r2 - a2) * k[i][j];
      double b2 = b - E2 - y1 * (r1 - a1) * k[i][j] - y2 * (r2 - a2) * k[j][j];

      if (0 < r1 && r1 < C) {
        b = b1;
      } else if (0 < r2 && r2 < C) {
        b = b2;
      } else {
        b = 0.5 * (b1 + b2);
      }

      a[i] = r1;
      a[j] = r2;
      changed = true;
    }

    if (changed) {
      passes = 0;
    } else {
      ++passes;
    }
  }

  return a;
}

void dump_array(const array& arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    std::cout << ' ' << arr[i];
  }
  std::cout << std::endl;
}

int main()
{
  array ys;
  std::vector<array> xs;
  freopen("small.txt", "r", stdin);

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream stream(line);

    double y;
    stream >> y;
    if (stream.peek() == ',')
      stream.ignore();

    ys.push_back(y);
    array row;

    double x;
    while (stream >> x) {
      row.push_back(x);

      if (stream.peek() == ',')
        stream.ignore();
    }
    xs.push_back(row);
  }
  std::cout << "read data" << std::endl;

  LinearKernel k;
  double b;
  array a = sequential_minimal_optimization(xs, ys, k, b, 1.0);
  std::cout << "result: b = " << b << "; x =";
  array x(xs[0].size(), 0.0);
  assert(xs.size() == a.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    x += a[i] * xs[i];
  }
  for (size_t i = 0; i < x.size(); ++i) {
    std::cout << ' ' << x[i];
  }
  std::cout << std::endl;

  return 0;
}

