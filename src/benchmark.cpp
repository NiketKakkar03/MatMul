#include "matmul/benchmark.hpp"
#include "build_info.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace matmul {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}
std::string timestamp() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm time{};
#ifdef _WIN32
    gmtime_s(&time, &now);
#else
    gmtime_r(&now, &time);
#endif
    std::ostringstream out;
    out << std::put_time(&time, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}
std::string json_quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
            else out << c;
        }
    }
    out << '"';
    return out.str();
}
std::string csv_quote(const std::string& value) {
    std::string result = "\"";
    for (char c : value) { result += c; if (c == '"') result += '"'; }
    return result + '"';
}
using Field = std::pair<std::string, std::string>;
std::vector<Field> fields(const BenchmarkResult& r, bool json) {
    std::vector<Field> result;
    const auto str = [&](std::string key, std::string value) {
        result.emplace_back(std::move(key), json ? json_quote(value) : csv_quote(value));
    };
    const auto number = [&](std::string key, auto value) {
        std::ostringstream out;
        out << std::setprecision(17);
        if (!std::isfinite(static_cast<double>(value))) out << (json ? "null" : "inf");
        else out << value;
        result.emplace_back(std::move(key), out.str());
    };
    const auto& c = r.config;
    str("implementation", c.implementation);
    number("matrix_size", c.rows_a == c.cols_a && c.cols_a == c.cols_b ? c.rows_a : 0);
    number("rows_a", c.rows_a); number("cols_a", c.cols_a); number("cols_b", c.cols_b);
    number("block_size", c.options.block_size); number("thread_count", r.thread_count);
    number("elapsed_time_ms", r.elapsed_time_ms); number("minimum_time_ms", r.minimum_time_ms);
    number("standard_deviation_ms", r.standard_deviation_ms); number("gflops", r.gflops);
    number("estimated_bandwidth", r.estimated_bandwidth);
    number("maximum_absolute_error", r.error.maximum_absolute_error);
    number("maximum_relative_error", r.error.maximum_relative_error);
    result.emplace_back("correct", r.error.correct ? "true" : "false");
    number("initialization_time_ms", r.initialization_time_ms);
    number("verification_time_ms", r.verification_time_ms); number("checksum", r.checksum);
    number("warmups", c.warmups); number("repetitions", c.repetitions); number("seed", c.seed);
    number("absolute_tolerance", c.absolute_tolerance); number("relative_tolerance", c.relative_tolerance);
    str("timestamp", r.timestamp); str("compiler", MATMUL_COMPILER);
    str("compiler_flags", MATMUL_FLAGS); str("machine_architecture", MATMUL_ARCH);
    str("build_type", MATMUL_BUILD_TYPE); number("logical_cpu_count", std::thread::hardware_concurrency());
    str("verification_reference", r.verification_reference);
    // Preserve every sample as a semicolon-separated string in the common schema.
    std::ostringstream durations;
    durations << std::setprecision(17);
    for (std::size_t i = 0; i < r.durations_ms.size(); ++i) {
        if (i) durations << ';';
        durations << r.durations_ms[i];
    }
    str("durations_ms", durations.str());
    return result;
}
}

BenchmarkResult run_benchmark(const BenchmarkConfig& config) {
    if (config.warmups == 0 || config.repetitions == 0)
        throw std::invalid_argument("benchmark requires at least one warmup and repetition");
    if (!std::isfinite(config.absolute_tolerance) || !std::isfinite(config.relative_tolerance) ||
        config.absolute_tolerance < 0 || config.relative_tolerance < 0)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    BenchmarkResult result;
    result.config = config;
    result.timestamp = timestamp();
    result.durations_ms.reserve(config.repetitions);
    const auto initialization_start = Clock::now();
    const auto a = Matrix::random(config.rows_a, config.cols_a, config.seed);
    const auto b = Matrix::random(config.cols_a, config.cols_b, config.seed + 1);
    Matrix c(config.rows_a, config.cols_b);
    result.initialization_time_ms = milliseconds(Clock::now() - initialization_start);
    if (config.implementation == "parallel" || config.implementation == "blocked_parallel")
        result.thread_count = resolved_thread_count(config.options.threads, config.rows_a);
    else if (config.implementation == "accelerate")
        result.thread_count = 0; // library-managed: never claim a controlled thread count
    for (std::size_t i = 0; i < config.warmups; ++i)
        multiply_into(a, b, c, config.implementation, config.options);
    for (std::size_t i = 0; i < config.repetitions; ++i) {
        const auto start = Clock::now();
        multiply_into(a, b, c, config.implementation, config.options);
        const auto stop = Clock::now();
        result.durations_ms.push_back(milliseconds(stop - start));
        // Observe every iteration outside timing; prevents dead result elimination.
        for (std::size_t j = 0; j < c.size(); ++j) result.checksum += c.data()[j];
    }
    auto sorted = result.durations_ms;
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    result.elapsed_time_ms = sorted.size() % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    result.minimum_time_ms = sorted.front();
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double variance = 0;
    for (double duration : sorted) variance += (duration - mean) * (duration - mean);
    result.standard_deviation_ms = std::sqrt(variance / sorted.size());
    const double seconds = result.elapsed_time_ms / 1000;
    if (seconds > 0) {
        result.gflops = 2.0 * static_cast<double>(a.rows()) * a.cols() * b.cols() / seconds / 1e9;
        if (c.size() != 0)
            result.estimated_bandwidth = (static_cast<double>(a.size()) + b.size() + c.size()) * sizeof(float) / seconds / 1e9;
    }
    const auto verification_start = Clock::now();
    const long double work = static_cast<long double>(a.rows()) * a.cols() * b.cols();
    Matrix expected(a.rows(), b.cols());
    if (accelerate_available() && config.implementation != "accelerate" && work > 16'777'216) {
        multiply_accelerate(a, b, expected);
        result.verification_reference = "Apple Accelerate FP32";
    } else {
        expected = reference_multiply(a, b);
        result.verification_reference = "independent FP64 dot products";
    }
    result.error = compare(c, expected, config.absolute_tolerance, config.relative_tolerance);
    result.verification_time_ms = milliseconds(Clock::now() - verification_start);
    return result;
}
void print_result(std::ostream& out, const BenchmarkResult& r) {
    out << "Implementation: " << r.config.implementation << "\nMatrix:         "
        << r.config.rows_a << " x " << r.config.cols_a << " times " << r.config.cols_a << " x " << r.config.cols_b
        << "\nBlock size:     " << r.config.options.block_size << "\nThreads:        ";
    if (r.config.implementation == "accelerate") out << "library-managed";
    else out << r.thread_count;
    out << std::setprecision(8) << "\nMedian time:    " << r.elapsed_time_ms
        << " ms\nMinimum time:   " << r.minimum_time_ms << " ms\nStd deviation:  " << r.standard_deviation_ms
        << " ms\nPerformance:    " << r.gflops << " GFLOP/s\nEffective I/O:  " << r.estimated_bandwidth
        << " GB/s (lower-bound traffic model)\nMax abs error:  " << r.error.maximum_absolute_error
        << "\nMax rel error:  " << r.error.maximum_relative_error << "\nCorrect:        "
        << (r.error.correct ? "yes" : "NO") << "\nChecksum:       " << r.checksum
        << "\nReference:      " << r.verification_reference << '\n';
}
void write_json(std::ostream& out, const BenchmarkResult& result) {
    out << '{';
    const auto values = fields(result, true);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << json_quote(values[i].first) << ':' << values[i].second;
    }
    out << "}\n";
}
void write_csv(const std::filesystem::path& path, const std::vector<BenchmarkResult>& results, bool append) {
    if (results.empty()) return;
    const auto values = fields(results.front(), false);
    std::ostringstream header;
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) header << ','; header << values[i].first; }
    bool existing = append && std::filesystem::exists(path) && std::filesystem::file_size(path) != 0;
    if (existing) {
        std::ifstream previous(path);
        std::string first_line;
        std::getline(previous, first_line);
        if (first_line != header.str()) throw std::runtime_error("CSV schema mismatch; use a new output file");
    }
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, append ? std::ios::app : std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open CSV: " + path.string());
    if (!existing) out << header.str() << '\n';
    for (const auto& result : results) {
        const auto row = fields(result, false);
        for (std::size_t i = 0; i < row.size(); ++i) { if (i) out << ','; out << row[i].second; }
        out << '\n';
    }
    if (!out) throw std::runtime_error("failed writing CSV: " + path.string());
}
bool run_verification(std::ostream& out) {
    bool passed = true;
    for (const auto& name : implementation_names()) {
        bool correct = true;
        for (const auto& shape : std::vector<std::vector<std::size_t>>{{1,1,1},{2,3,4},{7,9,5},{65,17,63},{0,3,4},{3,0,4}}) {
            const auto a = Matrix::random(shape[0], shape[1], 42);
            const auto b = Matrix::random(shape[1], shape[2], 43);
            const auto c = multiply(a, b, name, {8, 3});
            correct = compare(c, reference_multiply(a, b)).correct && correct;
        }
        out << name << ": " << (correct ? "PASS" : "FAIL") << '\n';
        passed = correct && passed;
    }
    return passed;
}
}
