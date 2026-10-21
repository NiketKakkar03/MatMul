#include "matmul/algorithms.hpp"
#include <stdexcept>

namespace matmul {

void multiply_reordered(const Matrix& a, const Matrix& b, Matrix& c) {
    multiply_loop_order(a, b, c, "ikj");
}

void multiply_loop_order(const Matrix& a, const Matrix& b, Matrix& c,
                         std::string_view order) {
    if (order == "i-j-k") order = "ijk";
    if (order == "i-k-j") order = "ikj";
    if (order == "j-i-k") order = "jik";
    if (order == "j-k-i") order = "jki";
    if (order == "k-i-j") order = "kij";
    if (order == "k-j-i") order = "kji";
    if (order == "ijk") {
        multiply_naive(a, b, c);
        return;
    }
    if (order != "ikj" && order != "jik" && order != "jki" &&
        order != "kij" && order != "kji") {
        throw std::invalid_argument("Unknown loop order: " + std::string(order));
    }
    validate_multiplication(a, b, c);
    c.fill(0.0F);
    const auto m = a.rows(), n = b.cols(), inner = a.cols();
    if (m == 0 || n == 0 || inner == 0) return;
    const float* ap = a.data();
    const float* bp = b.data();
    float* cp = c.data();
    if (order == "ikj") {
        for (std::size_t i = 0; i < m; ++i) {
            const float* ar = ap + i * inner;
            float* cr = cp + i * n;
            for (std::size_t k = 0; k < inner; ++k) {
                const float aik = ar[k];
                const float* br = bp + k * n;
                for (std::size_t j = 0; j < n; ++j) cr[j] += aik * br[j];
            }
        }
    } else if (order == "jik") {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                float sum = 0.0F;
                for (std::size_t k = 0; k < inner; ++k) {
                    sum += ap[i * inner + k] * bp[k * n + j];
                }
                cp[i * n + j] = sum;
            }
        }
    } else if (order == "jki") {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < inner; ++k) {
                const float bkj = bp[k * n + j];
                for (std::size_t i = 0; i < m; ++i) {
                    cp[i * n + j] += ap[i * inner + k] * bkj;
                }
            }
        }
    } else if (order == "kij") {
        for (std::size_t k = 0; k < inner; ++k) {
            const float* br = bp + k * n;
            for (std::size_t i = 0; i < m; ++i) {
                const float aik = ap[i * inner + k];
                float* cr = cp + i * n;
                for (std::size_t j = 0; j < n; ++j) cr[j] += aik * br[j];
            }
        }
    } else {
        for (std::size_t k = 0; k < inner; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                const float bkj = bp[k * n + j];
                for (std::size_t i = 0; i < m; ++i) {
                    cp[i * n + j] += ap[i * inner + k] * bkj;
                }
            }
        }
    }
}

} // namespace matmul
