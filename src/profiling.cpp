#include "matmul/profiling.hpp"
#include <chrono>
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace matmul {
ProfileResult run_profile_workload(const BenchmarkConfig& config, double duration_seconds,
                                   bool wait_for_profiler, std::istream& input, std::ostream& output) {
    if (!std::isfinite(duration_seconds) || duration_seconds <= 0)
        throw std::invalid_argument("profile duration must be finite and positive");
    if (config.warmups == 0)
        throw std::invalid_argument("profile requires at least one warmup");
    const auto a = Matrix::random(config.rows_a, config.cols_a, config.seed);
    const auto b = Matrix::random(config.cols_a, config.cols_b, config.seed + 1);
    Matrix c(config.rows_a, config.cols_b);
    for (std::size_t i = 0; i < config.warmups; ++i)
        multiply_into(a, b, c, config.implementation, config.options);
    if (wait_for_profiler) {
        output << "PROFILE_READY\n" << std::flush;
        std::string signal;
        if (!std::getline(input, signal))
            throw std::runtime_error("profiler handshake ended before the start signal");
    }
    ProfileResult result;
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    do {
        multiply_into(a, b, c, config.implementation, config.options);
        ++result.iterations;
    } while (std::chrono::duration<double>(Clock::now() - start).count() < duration_seconds);
    result.elapsed_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (std::size_t i = 0; i < c.size(); ++i) result.checksum += c.data()[i];
    return result;
}
}
