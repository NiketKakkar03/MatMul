#pragma once
#include "matmul/algorithms.hpp"
#include "matmul/verification.hpp"
#include <filesystem>
#include <iosfwd>

namespace matmul {
struct BenchmarkConfig {
    std::string implementation = "naive";
    std::size_t rows_a = 512;
    std::size_t cols_a = 512;
    std::size_t cols_b = 512;
    MultiplyOptions options;
    std::size_t warmups = 2;
    std::size_t repetitions = 9;
    std::uint32_t seed = 42;
    double absolute_tolerance = 1e-4;
    double relative_tolerance = 1e-4;
};
struct BenchmarkResult {
    BenchmarkConfig config;
    std::vector<double> durations_ms;
    double elapsed_time_ms = 0;
    double minimum_time_ms = 0;
    double standard_deviation_ms = 0;
    double gflops = 0;
    // Algorithm-independent lower bound: one read of A/B and one write of C.
    // Effective GB/s, NOT measured DRAM traffic or hardware bandwidth.
    double estimated_bandwidth = 0;
    double initialization_time_ms = 0;
    double verification_time_ms = 0;
    double checksum = 0;
    ErrorMetrics error;
    std::size_t thread_count = 1;
    std::string verification_reference;
    std::string timestamp;
};
[[nodiscard]] BenchmarkResult run_benchmark(const BenchmarkConfig& config);
void print_result(std::ostream& out, const BenchmarkResult& result);
void write_csv(const std::filesystem::path& path, const std::vector<BenchmarkResult>& results,
               bool append = false);
void write_json(std::ostream& out, const BenchmarkResult& result);
[[nodiscard]] bool run_verification(std::ostream& out);
}
