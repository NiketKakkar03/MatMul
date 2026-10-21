#pragma once
#include "matmul/matrix.hpp"
#include <string>
#include <string_view>
#include <vector>

namespace matmul {
struct MultiplyOptions {
    std::size_t block_size = 64;
    // Zero means auto. Nonparallel implementations ignore this field.
    std::size_t threads = 0;
};
// All kernels overwrite C, accept rectangular and empty shapes, reject shape
// mismatches and output aliasing. Only multiply() allocates its result.
void validate_multiplication(const Matrix& a, const Matrix& b, const Matrix& c);
void multiply_naive(const Matrix& a, const Matrix& b, Matrix& c);
void multiply_reordered(const Matrix& a, const Matrix& b, Matrix& c);
void multiply_loop_order(const Matrix& a, const Matrix& b, Matrix& c, std::string_view order);
void multiply_blocked(const Matrix& a, const Matrix& b, Matrix& c, std::size_t block_size);
void multiply_parallel(const Matrix& a, const Matrix& b, Matrix& c,
                       std::size_t block_size, std::size_t threads);
void multiply_accelerate(const Matrix& a, const Matrix& b, Matrix& c);
[[nodiscard]] bool accelerate_available() noexcept;
[[nodiscard]] std::size_t resolved_thread_count(std::size_t requested, std::size_t rows) noexcept;
[[nodiscard]] std::vector<std::string> implementation_names();
void multiply_into(const Matrix& a, const Matrix& b, Matrix& c,
                   std::string_view implementation, MultiplyOptions options = {});
[[nodiscard]] Matrix multiply(const Matrix& a, const Matrix& b,
                              std::string_view implementation = "naive", MultiplyOptions options = {});
}
