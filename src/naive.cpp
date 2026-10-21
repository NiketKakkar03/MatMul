#include "matmul/algorithms.hpp"

namespace matmul {

// Frozen i-j-k baseline: preserve the straightforward scalar accumulation.
void multiply_naive(const Matrix& a, const Matrix& b, Matrix& c) {
    validate_multiplication(a, b, c);
    if (a.rows() == 0 || b.cols() == 0) return;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            float sum = 0.0F;
            for (std::size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
}

} // namespace matmul
