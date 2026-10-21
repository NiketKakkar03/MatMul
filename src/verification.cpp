#include "matmul/verification.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace matmul {
ErrorMetrics compare(const Matrix& actual, const Matrix& expected, double atol, double rtol) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols())
        throw std::invalid_argument("comparison requires matching shapes");
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    ErrorMetrics result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = actual.data()[i], b = expected.data()[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            result.correct = false;
            result.maximum_absolute_error = result.maximum_relative_error = std::numeric_limits<double>::infinity();
            continue;
        }
        const double absolute = std::abs(a - b);
        const double scale = std::max(std::abs(a), std::abs(b));
        result.maximum_absolute_error = std::max(result.maximum_absolute_error, absolute);
        result.maximum_relative_error = std::max(result.maximum_relative_error, scale == 0 ? 0 : absolute / scale);
        if (absolute > atol + rtol * scale) result.correct = false;
    }
    return result;
}
bool approximately_equal(const Matrix& a, const Matrix& b, double atol, double rtol) {
    return compare(a, b, atol, rtol).correct;
}
Matrix reference_multiply(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) throw std::invalid_argument("incompatible multiplication shapes");
    Matrix result(a.rows(), b.cols());
    if (a.rows() == 0 || b.cols() == 0 || a.cols() == 0) return result;
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < b.cols(); ++j) {
            double sum = 0;
            for (std::size_t k = 0; k < a.cols(); ++k)
                sum += static_cast<double>(a(i, k)) * static_cast<double>(b(k, j));
            result(i, j) = static_cast<float>(sum);
        }
    return result;
}
}
