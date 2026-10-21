#pragma once

#include "matmul/benchmark.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tests {
using Test = std::pair<std::string, void (*)()>;
inline std::vector<Test>& registry() {
    static std::vector<Test> tests;
    return tests;
}
struct Register {
    Register(const char* name, void (*function)()) { registry().emplace_back(name, function); }
};
inline void require(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 ": requirement failed: " + expression);
    }
}
template <typename Exception, typename Function>
void require_throws(Function&& function, const char* expression, const char* file, int line) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, expression, file, line);
}
inline void require_equal(const matmul::Matrix& actual,
                          const matmul::Matrix& expected,
                          double tolerance = 1e-4) {
    if (!matmul::approximately_equal(actual, expected, tolerance, tolerance)) {
        const auto error = matmul::compare(actual, expected, tolerance, tolerance);
        throw std::runtime_error("matrix mismatch; max absolute error=" +
                                 std::to_string(error.maximum_absolute_error));
    }
}
} // namespace tests

#define MF_TEST(name) \
    static void name(); \
    static tests::Register register_##name(#name, name); \
    static void name()
#define MF_REQUIRE(expression) \
    tests::require(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
#define MF_THROWS(exception, expression) \
    tests::require_throws<exception>([&] { static_cast<void>(expression); }, #expression, __FILE__, __LINE__)
