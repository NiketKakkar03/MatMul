#include "test_support.hpp"
#include <chrono>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>

using namespace matmul;

namespace {
BenchmarkConfig tiny_config() {
    BenchmarkConfig config;
    config.rows_a = 9;
    config.cols_a = 7;
    config.cols_b = 5;
    config.implementation = "blocked";
    config.options = {4, 2};
    config.warmups = 1;
    config.repetitions = 5;
    return config;
}
bool near(double a, double b) {
    return std::abs(a - b) <= 1e-10 * std::max({1.0, std::abs(a), std::abs(b)});
}
std::size_t csv_field_count(const std::string& line) {
    std::size_t fields = 1;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') ++i;
            else quoted = !quoted;
        } else if (line[i] == ',' && !quoted) {
            ++fields;
        }
    }
    MF_REQUIRE(!quoted);
    return fields;
}
struct TemporaryCsv {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("matmul-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv");
    ~TemporaryCsv() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};
}

MF_TEST(benchmark_measures_repetitions_and_statistics) {
    const auto config = tiny_config();
    const auto result = run_benchmark(config);
    MF_REQUIRE(result.durations_ms.size() == config.repetitions);
    for (const double duration : result.durations_ms)
        MF_REQUIRE(std::isfinite(duration) && duration >= 0);
    auto sorted = result.durations_ms;
    std::sort(sorted.begin(), sorted.end());
    MF_REQUIRE(near(result.elapsed_time_ms, sorted[sorted.size() / 2]));
    MF_REQUIRE(near(result.minimum_time_ms, sorted.front()));
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
                        static_cast<double>(sorted.size());
    double variance = 0;
    for (const auto duration : sorted) variance += (duration - mean) * (duration - mean);
    variance /= static_cast<double>(sorted.size());
    MF_REQUIRE(near(result.standard_deviation_ms, std::sqrt(variance)));
    MF_REQUIRE(result.error.correct);
    MF_REQUIRE(!result.timestamp.empty());
    MF_REQUIRE(!result.verification_reference.empty());
    MF_REQUIRE(std::isfinite(result.checksum));
    MF_REQUIRE(result.initialization_time_ms >= 0);
    MF_REQUIRE(result.verification_time_ms >= 0);
    MF_REQUIRE(result.thread_count == 1);
    if (result.elapsed_time_ms > 0) {
        MF_REQUIRE(near(result.gflops, 2.0 * 9 * 7 * 5 / (result.elapsed_time_ms * 1e6)));
        MF_REQUIRE(near(result.estimated_bandwidth,
                        sizeof(float) * (9.0 * 7 + 7.0 * 5 + 9.0 * 5) /
                        (result.elapsed_time_ms * 1e6)));
    }
    auto repeat_config = config;
    repeat_config.repetitions = 2;
    const auto repeat = run_benchmark(repeat_config);
    MF_REQUIRE(near(result.checksum / static_cast<double>(config.repetitions),
                    repeat.checksum / static_cast<double>(repeat_config.repetitions)));
    MF_REQUIRE(near(repeat.elapsed_time_ms,
                    (repeat.durations_ms[0] + repeat.durations_ms[1]) / 2));
    repeat_config.implementation = "parallel";
    MF_REQUIRE(run_benchmark(repeat_config).thread_count == 2);
}

MF_TEST(benchmark_validation) {
    auto config = tiny_config();
    config.repetitions = 0;
    MF_THROWS(std::invalid_argument, run_benchmark(config));
    config = tiny_config();
    config.implementation = "unknown";
    MF_THROWS(std::invalid_argument, run_benchmark(config));
    config = tiny_config();
    config.options.block_size = 0;
    MF_THROWS(std::invalid_argument, run_benchmark(config));
    config = tiny_config();
    config.absolute_tolerance = -1;
    MF_THROWS(std::invalid_argument, run_benchmark(config));
    config = tiny_config();
    config.relative_tolerance = std::numeric_limits<double>::infinity();
    MF_THROWS(std::invalid_argument, run_benchmark(config));
}

MF_TEST(benchmark_empty_and_maximum_extents) {
    auto config = tiny_config();
    config.rows_a = std::numeric_limits<std::size_t>::max();
    config.cols_a = 0;
    config.cols_b = 0;
    config.repetitions = 1;
    const auto huge_tall = run_benchmark(config);
    MF_REQUIRE(huge_tall.error.correct);
    MF_REQUIRE(huge_tall.checksum == 0 && huge_tall.gflops == 0);
    MF_REQUIRE(huge_tall.estimated_bandwidth == 0);
    config.rows_a = 0;
    config.cols_b = std::numeric_limits<std::size_t>::max();
    const auto huge_wide = run_benchmark(config);
    MF_REQUIRE(huge_wide.error.correct);
    MF_REQUIRE(huge_wide.checksum == 0 && huge_wide.gflops == 0);
    MF_REQUIRE(huge_wide.estimated_bandwidth == 0);
    // A nonempty unused input still causes no matrix traffic in an empty product.
    config.cols_a = 3;
    config.cols_b = 4;
    const auto empty_output = run_benchmark(config);
    MF_REQUIRE(empty_output.error.correct);
    MF_REQUIRE(empty_output.checksum == 0 && empty_output.gflops == 0);
    MF_REQUIRE(empty_output.estimated_bandwidth == 0);
}

MF_TEST(benchmark_json_csv_and_human_output) {
    const auto result = run_benchmark(tiny_config());
    std::ostringstream json;
    write_json(json, result);
    const auto encoded = json.str();
    MF_REQUIRE(encoded.find('{') != std::string::npos);
    MF_REQUIRE(encoded.find('}') != std::string::npos);
    MF_REQUIRE(encoded.find("\"implementation\"") != std::string::npos);
    MF_REQUIRE(encoded.find("\"elapsed_time_ms\"") != std::string::npos);
    MF_REQUIRE(encoded.find("blocked") != std::string::npos);
    MF_REQUIRE(encoded.find("nan") == std::string::npos);
    MF_REQUIRE(encoded.find("infinity") == std::string::npos);
    std::ostringstream human;
    print_result(human, result);
    MF_REQUIRE(human.str().find("blocked") != std::string::npos);
    MF_REQUIRE(!human.str().empty());
    TemporaryCsv csv;
    write_csv(csv.path, {result});
    write_csv(csv.path, {result}, true);
    std::ifstream input(csv.path);
    MF_REQUIRE(input.good());
    std::vector<std::string> lines;
    for (std::string line; std::getline(input, line);)
        if (!line.empty()) lines.push_back(line);
    MF_REQUIRE(lines.size() == 3);
    MF_REQUIRE(lines[0].find("implementation") != std::string::npos);
    MF_REQUIRE(lines[0].find("elapsed_time_ms") != std::string::npos);
    MF_REQUIRE(lines[1] == lines[2]);
    MF_REQUIRE(csv_field_count(lines[0]) == csv_field_count(lines[1]));
    write_csv(csv.path, {result});
    std::ifstream replaced(csv.path);
    std::string replaced_contents((std::istreambuf_iterator<char>(replaced)), {});
    MF_REQUIRE(std::count(replaced_contents.begin(), replaced_contents.end(), '\n') == 2);
}
