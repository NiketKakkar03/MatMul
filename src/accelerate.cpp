#include "matmul/algorithms.hpp"
#include <limits>
#include <stdexcept>

#if defined(MATMUL_HAS_ACCELERATE) && MATMUL_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace matmul {

bool accelerate_available() noexcept {
#if defined(MATMUL_HAS_ACCELERATE) && MATMUL_HAS_ACCELERATE
    return true;
#else
    return false;
#endif
}

void multiply_accelerate(const Matrix& a, const Matrix& b, Matrix& c) {
    validate_multiplication(a, b, c);
#if defined(MATMUL_HAS_ACCELERATE) && MATMUL_HAS_ACCELERATE
    const auto m = a.rows(), n = b.cols(), inner = a.cols();
    // BLAS does not need to inspect empty buffers, and a zero inner dimension
    // must still overwrite an existing output matrix.
    if (m == 0 || n == 0 || inner == 0) {
        c.fill(0.0F);
        return;
    }
    constexpr auto max_blas_dimension = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (m > max_blas_dimension || n > max_blas_dimension || inner > max_blas_dimension) {
        throw std::overflow_error("Matrix dimensions exceed the Accelerate BLAS integer range");
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(inner),
                1.0F, a.data(), static_cast<int>(inner), b.data(), static_cast<int>(n),
                0.0F, c.data(), static_cast<int>(n));
#else
    throw std::runtime_error("Apple Accelerate is unavailable in this build");
#endif
}

} // namespace matmul
