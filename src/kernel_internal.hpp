#pragma once

#include "matmul/matrix.hpp"
#include <algorithm>
#include <cstddef>

namespace matmul::detail {

// The caller owns [first_row, last_row), validates shapes and clears C.
// Taking the remaining extent before adding also accepts SIZE_MAX block sizes.
inline void blocked_rows(const Matrix& a, const Matrix& b, Matrix& c,
                         std::size_t block_size, std::size_t first_row,
                         std::size_t last_row) noexcept {
    const auto inner = a.cols();
    const auto columns = b.cols();
    const float* ap = a.data();
    const float* bp = b.data();
    float* cp = c.data();
    for (auto ii = first_row; ii < last_row;) {
        const auto i_end = ii + std::min(block_size, last_row - ii);
        for (std::size_t kk = 0; kk < inner;) {
            const auto k_end = kk + std::min(block_size, inner - kk);
            for (std::size_t jj = 0; jj < columns;) {
                const auto j_end = jj + std::min(block_size, columns - jj);
                for (auto i = ii; i < i_end; ++i) {
                    const float* ar = ap + i * inner;
                    float* cr = cp + i * columns;
                    for (auto k = kk; k < k_end; ++k) {
                        const float aik = ar[k];
                        const float* br = bp + k * columns;
                        for (auto j = jj; j < j_end; ++j) {
                            cr[j] += aik * br[j];
                        }
                    }
                }
                jj = j_end;
            }
            kk = k_end;
        }
        ii = i_end;
    }
}

} // namespace matmul::detail
