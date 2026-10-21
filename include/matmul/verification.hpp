#pragma once
#include "matmul/matrix.hpp"

namespace matmul {
struct ErrorMetrics {
    double maximum_absolute_error = 0;
    double maximum_relative_error = 0;
    bool correct = true;
};
// Symmetric relative error; denominator is max(abs(actual), abs(expected)).
// Nonfinite values always fail verification, including matching infinities.
[[nodiscard]] ErrorMetrics compare(const Matrix& actual, const Matrix& expected,
                                   double absolute_tolerance = 1e-4,
                                   double relative_tolerance = 1e-4);
[[nodiscard]] bool approximately_equal(const Matrix& a, const Matrix& b,
                                       double atol = 1e-4, double rtol = 1e-4);
// Independent high precision oracle, rounding only the final dot product to FP32.
[[nodiscard]] Matrix reference_multiply(const Matrix& a, const Matrix& b);
}
