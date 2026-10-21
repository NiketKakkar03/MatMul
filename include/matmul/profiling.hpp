#pragma once
#include "matmul/benchmark.hpp"
#include <iosfwd>

namespace matmul {
struct ProfileResult {
    std::size_t iterations = 0;
    double elapsed_time_ms = 0;
    double checksum = 0;
};
// Sustained kernel workload for external sampling. Input preparation and
// warmups precede readiness; no verification oracle runs in the sample window.
[[nodiscard]] ProfileResult run_profile_workload(const BenchmarkConfig& config,
    double duration_seconds, bool wait_for_profiler, std::istream& input, std::ostream& output);
}
