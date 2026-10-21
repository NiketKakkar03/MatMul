#include "matmul/benchmark.hpp"
#include "matmul/profiling.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
using namespace matmul;
void help() {
    std::cout << R"(MatMul — C++20 FP32 matrix multiplication experiments

matmul benchmark [options]
matmul sweep [options] --output results/results.csv
matmul verify
matmul list [--json]
matmul profile [shape/kernel options] --duration 5

Benchmark options:
  --implementation NAME    naive, reordered, ijk, ikj, jik, jki, kij, kji,
                           blocked, parallel (alias blocked_parallel), accelerate
  --size N                 Square size (default 512)
  --rows M --inner K --cols N  Rectangular A(M,K) * B(K,N)
  --block-size N           Tile edge (default 64, must be positive)
  --threads auto|N         Parallel workers (default auto); BLAS manages its own
  --warmups N              Warmups (default 2, at least 1)
  --repetitions N          Measured iterations (default 9, at least 1)
  --seed N                 Input seed (default 42)
  --atol X --rtol X        Comparison tolerances (default 1e-4 each)
  --output FILE.csv        Write result; --append appends compatible CSV
  --json                   Emit one JSON result on stdout

Sweep options (also accepts common benchmark options):
  --sizes 32,64,...        Default 32,64,128,256,512,768,1024,1536,2048
  --implementations LIST  Default all canonical implementations
  --max-seconds X         Overall budget, checked between configurations (default 300)

Use scripts/run_sweep.py for timeout-isolated full block/thread/shape studies.
Profile runs kernels without the verification oracle. --wait-for-profiler emits
PROFILE_READY after setup/warmups, then waits for a newline before kernel work.
Release builds are required for meaningful timing. Empty shapes are legal.
)";
}
using Options = std::map<std::string, std::string>;
Options parse(int argc, char** argv) {
    const std::set<std::string> flags{"--json", "--append", "--help", "--wait-for-profiler"};
    const std::set<std::string> valued{"--implementation", "--size", "--rows", "--inner", "--cols",
        "--block-size", "--threads", "--warmups", "--repetitions", "--seed", "--output", "--atol", "--rtol",
        "--sizes", "--implementations", "--max-seconds", "--duration"};
    Options result;
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (result.contains(key)) throw std::invalid_argument("duplicate option: " + key);
        if (flags.contains(key)) result[key] = "true";
        else if (valued.contains(key)) {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
            result[key] = argv[++i];
        } else throw std::invalid_argument("unknown option: " + key);
    }
    return result;
}
std::size_t integer(const std::string& value) {
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument("expected an unsigned integer: " + value);
    return result;
}
double real(const std::string& value) {
    std::size_t consumed = 0;
    const double result = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result) || result < 0)
        throw std::invalid_argument("expected a finite nonnegative number: " + value);
    return result;
}
std::vector<std::string> split(const std::string& value) {
    std::vector<std::string> result;
    std::istringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::invalid_argument("empty list entry");
        result.push_back(item);
    }
    if (result.empty() || value.back() == ',') throw std::invalid_argument("empty list entry");
    return result;
}
BenchmarkConfig configuration(const Options& options) {
    BenchmarkConfig c;
    if (options.contains("--size")) {
        if (options.contains("--rows") || options.contains("--inner") || options.contains("--cols"))
            throw std::invalid_argument("use --size or all three rectangular dimensions");
        c.rows_a = c.cols_a = c.cols_b = integer(options.at("--size"));
    }
    const int dimensions = options.contains("--rows") + options.contains("--inner") + options.contains("--cols");
    if (dimensions != 0 && dimensions != 3) throw std::invalid_argument("rectangular shape requires --rows, --inner, --cols");
    if (dimensions == 3) {
        c.rows_a = integer(options.at("--rows")); c.cols_a = integer(options.at("--inner")); c.cols_b = integer(options.at("--cols"));
    }
    if (options.contains("--implementation")) c.implementation = options.at("--implementation");
    if (options.contains("--block-size")) c.options.block_size = integer(options.at("--block-size"));
    if (c.options.block_size == 0) throw std::invalid_argument("block size must be positive");
    if (options.contains("--threads") && options.at("--threads") != "auto") {
        c.options.threads = integer(options.at("--threads"));
        if (!c.options.threads) throw std::invalid_argument("threads must be positive or auto");
    }
    if (options.contains("--warmups")) c.warmups = integer(options.at("--warmups"));
    if (options.contains("--repetitions")) c.repetitions = integer(options.at("--repetitions"));
    if (options.contains("--seed")) {
        const auto seed = integer(options.at("--seed"));
        if (seed > UINT32_MAX) throw std::invalid_argument("seed exceeds uint32 range");
        c.seed = static_cast<std::uint32_t>(seed);
    }
    if (options.contains("--atol")) c.absolute_tolerance = real(options.at("--atol"));
    if (options.contains("--rtol")) c.relative_tolerance = real(options.at("--rtol"));
    return c;
}
void validate_options(const Options& options, const std::set<std::string>& allowed,
                      const std::string& command) {
    for (const auto& [key, value] : options) {
        (void)value;
        if (!allowed.contains(key)) throw std::invalid_argument("unsupported option for " + command + ": " + key);
    }
}
}
int main(int argc, char** argv) {
    try {
        using namespace matmul;
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") { help(); return 0; }
        const std::string command = argv[1];
        const auto options = parse(argc, argv);
        if (options.contains("--help")) { help(); return 0; }
        if (command == "list") {
            if (options.size() > (options.contains("--json") ? 1U : 0U)) throw std::invalid_argument("list only supports --json");
            const auto names = implementation_names();
            if (options.contains("--json")) std::cout << '[';
            for (std::size_t i = 0; i < names.size(); ++i) {
                if (options.contains("--json")) { if (i) std::cout << ','; std::cout << '"' << names[i] << '"'; }
                else std::cout << names[i] << '\n';
            }
            if (options.contains("--json")) std::cout << "]\n";
            return 0;
        }
        if (command == "verify") {
            if (!options.empty()) throw std::invalid_argument("verify accepts no options");
            return run_verification(std::cout) ? 0 : 1;
        }
        if (command == "profile") {
            validate_options(options, {"--implementation", "--size", "--rows", "--inner", "--cols",
                "--block-size", "--threads", "--warmups", "--seed", "--duration", "--wait-for-profiler"}, command);
            const auto config = configuration(options);
            const double duration = options.contains("--duration") ? real(options.at("--duration")) : 5;
            const auto result = run_profile_workload(config, duration, options.contains("--wait-for-profiler"), std::cin, std::cout);
            std::cout << std::setprecision(17) << "{\"implementation\":\"" << config.implementation
                << "\",\"iterations\":" << result.iterations << ",\"elapsed_time_ms\":" << result.elapsed_time_ms
                << ",\"checksum\":" << result.checksum << "}\n";
            return 0;
        }
        if (command != "benchmark" && command != "sweep") throw std::invalid_argument("unknown command: " + command);
        validate_options(options, {"--implementation", "--size", "--rows", "--inner", "--cols",
            "--block-size", "--threads", "--warmups", "--repetitions", "--seed", "--output", "--append",
            "--json", "--atol", "--rtol", "--sizes", "--implementations", "--max-seconds"}, command);
        auto config = configuration(options);
        if (command == "benchmark") {
            if (options.contains("--append") && !options.contains("--output"))
                throw std::invalid_argument("--append requires --output");
            if (options.contains("--sizes") || options.contains("--implementations") || options.contains("--max-seconds"))
                throw std::invalid_argument("sweep options are not supported by benchmark");
            const auto result = run_benchmark(config);
            if (options.contains("--output")) write_csv(options.at("--output"), {result}, options.contains("--append"));
            if (options.contains("--json")) write_json(std::cout, result); else print_result(std::cout, result);
            return result.error.correct ? 0 : 1;
        }
        if (options.contains("--size") || options.contains("--rows") || options.contains("--implementation"))
            throw std::invalid_argument("sweep uses --sizes and --implementations; rectangular sweeps use scripts/run_sweep.py");
        auto sizes = split(options.contains("--sizes") ? options.at("--sizes") : "32,64,128,256,512,768,1024,1536,2048");
        auto names = options.contains("--implementations") ? split(options.at("--implementations")) : implementation_names();
        std::vector<BenchmarkConfig> jobs;
        for (const auto& size : sizes) for (const auto& name : names) {
            auto job = config;
            job.rows_a = job.cols_a = job.cols_b = integer(size); job.implementation = name;
            jobs.push_back(job);
        }
        std::mt19937 random(config.seed);
        std::shuffle(jobs.begin(), jobs.end(), random);
        const double budget = options.contains("--max-seconds") ? real(options.at("--max-seconds")) : 300;
        const auto start = std::chrono::steady_clock::now();
        const std::string output = options.contains("--output") ? options.at("--output") : "results/results.csv";
        bool append = options.contains("--append"), correct = true;
        std::size_t completed = 0;
        for (const auto& job : jobs) {
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= budget) {
                std::cerr << "Sweep budget reached; skipped " << jobs.size() - completed << " configurations\n";
                break;
            }
            auto result = run_benchmark(job);
            write_csv(output, {result}, append); append = true;
            if (options.contains("--json")) write_json(std::cout, result); else print_result(std::cout, result);
            correct = result.error.correct && correct;
            ++completed;
        }
        std::cerr << "Wrote " << completed << " configurations to " << output << '\n';
        return correct && completed > 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "matmul: " << error.what() << '\n';
        return 2;
    }
}
