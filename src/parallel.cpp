#include "matmul/algorithms.hpp"
#include "kernel_internal.hpp"
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>

namespace matmul {

std::size_t resolved_thread_count(std::size_t requested, std::size_t rows) noexcept {
    if (rows == 0) return 0;
    if (requested == 0) {
        requested = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (requested == 0) requested = 1;
    }
    return std::min(requested, rows);
}

void multiply_parallel(const Matrix& a, const Matrix& b, Matrix& c,
                       std::size_t block_size, std::size_t threads) {
    validate_multiplication(a, b, c);
    if (block_size == 0) throw std::invalid_argument("Block size must be positive");
    c.fill(0.0F);
    if (a.rows() == 0 || b.cols() == 0 || a.cols() == 0) return;
    const auto count = resolved_thread_count(threads, a.rows());
    if (count == 1) {
        detail::blocked_rows(a, b, c, block_size, 0, a.rows());
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(count - 1);
    // Declared after workers, so every started thread is joined before its
    // std::thread is destroyed, including if a later creation throws.
    struct JoinWorkers {
        std::vector<std::thread>& workers;
        ~JoinWorkers() {
            for (auto& worker : workers) {
                if (worker.joinable()) worker.join();
            }
        }
    } join_workers{workers};

    const auto rows_per_worker = a.rows() / count;
    const auto remainder = a.rows() % count;
    std::size_t first = 0;
    for (std::size_t worker = 0; worker + 1 < count; ++worker) {
        const auto last = first + rows_per_worker + (worker < remainder ? 1 : 0);
        workers.emplace_back([&a, &b, &c, block_size, first, last] {
            detail::blocked_rows(a, b, c, block_size, first, last);
        });
        first = last;
    }
    detail::blocked_rows(a, b, c, block_size, first, a.rows());
}

} // namespace matmul
