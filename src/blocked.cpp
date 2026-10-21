#include "matmul/algorithms.hpp"
#include "kernel_internal.hpp"
#include <stdexcept>

namespace matmul {

void multiply_blocked(const Matrix& a, const Matrix& b, Matrix& c,
                      std::size_t block_size) {
    validate_multiplication(a, b, c);
    if (block_size == 0) throw std::invalid_argument("Block size must be positive");
    c.fill(0.0F);
    if (a.rows() == 0 || b.cols() == 0 || a.cols() == 0) return;
    detail::blocked_rows(a, b, c, block_size, 0, a.rows());
}

} // namespace matmul
