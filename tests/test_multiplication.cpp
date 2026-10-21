#include "test_support.hpp"
#include <limits>

using namespace matmul;

namespace {
std::vector<std::string> portable_implementations() {
    std::vector<std::string> names{"naive", "reordered", "blocked", "parallel"};
    if (accelerate_available()) names.emplace_back("accelerate");
    return names;
}
}

MF_TEST(hand_computed_rectangular_product) {
    Matrix a(2, 3);
    Matrix b(3, 2);
    for (std::size_t i = 0; i < a.size(); ++i) a.data()[i] = static_cast<float>(i + 1);
    for (std::size_t i = 0; i < b.size(); ++i) b.data()[i] = static_cast<float>(i + 7);
    Matrix expected(2, 2);
    expected(0, 0) = 58; expected(0, 1) = 64;
    expected(1, 0) = 139; expected(1, 1) = 154;
    tests::require_equal(reference_multiply(a, b), expected, 0);
    for (const auto& implementation : portable_implementations()) {
        Matrix output(2, 2, 999);
        const float* storage = output.data();
        multiply_into(a, b, output, implementation, {2, 2});
        MF_REQUIRE(output.data() == storage);
        tests::require_equal(output, expected, 0);
        multiply_into(a, b, output, implementation, {2, 2});
        tests::require_equal(output, expected, 0);
        tests::require_equal(multiply(a, b, implementation, {2, 2}), expected, 0);
    }
}

MF_TEST(all_six_loop_orders) {
    const auto a = Matrix::random(7, 11, 42);
    const auto b = Matrix::random(11, 5, 43);
    const auto expected = reference_multiply(a, b);
    for (const auto order : {"ijk", "ikj", "jik", "jki", "kij", "kji"}) {
        Matrix output(7, 5, 999);
        multiply_loop_order(a, b, output, order);
        tests::require_equal(output, expected);
    }
}

MF_TEST(identity_and_zero_products) {
    const auto a = Matrix::random(5, 8, 101);
    for (const auto& implementation : portable_implementations()) {
        tests::require_equal(multiply(a, Matrix::identity(8), implementation), a, 0);
        tests::require_equal(multiply(Matrix::identity(5), a, implementation), a, 0);
        tests::require_equal(multiply(a, Matrix(8, 3), implementation), Matrix(5, 3), 0);
        tests::require_equal(multiply(Matrix(4, 5), a, implementation), Matrix(4, 8), 0);
    }
}

MF_TEST(blocked_tails_and_rectangular_large_dimensions) {
    for (const std::size_t size : {63U, 65U, 127U, 129U}) {
        const auto a = Matrix::random(size, size, 10);
        const auto b = Matrix::random(size, size, 11);
        const auto expected = reference_multiply(a, b);
        Matrix output(size, size, 555);
        for (const std::size_t block : {16U, 64U}) {
            multiply_blocked(a, b, output, block);
            tests::require_equal(output, expected);
            multiply_parallel(a, b, output, block, 4);
            tests::require_equal(output, expected);
        }
    }
    for (const std::size_t size : {1000U, 1025U}) {
        const auto a = Matrix::random(size, 3, 1);
        const auto b = Matrix::random(3, 2, 2);
        const auto expected = reference_multiply(a, b);
        Matrix output(size, 2);
        multiply_blocked(a, b, output, 64);
        tests::require_equal(output, expected);
        multiply_parallel(a, b, output, 64, 4);
        tests::require_equal(output, expected);
        const auto wide_a = Matrix::random(2, size, 3);
        const auto wide_b = Matrix::random(size, 3, 4);
        tests::require_equal(multiply(wide_a, wide_b, "blocked"),
                             reference_multiply(wide_a, wide_b));
    }
}

MF_TEST(thread_counts_and_small_row_partitions) {
    const auto a = Matrix::random(3, 17, 44);
    const auto b = Matrix::random(17, 9, 45);
    const auto expected = reference_multiply(a, b);
    Matrix output(3, 9);
    for (const std::size_t threads : {0U, 1U, 2U, 4U, 99U}) {
        multiply_parallel(a, b, output, 4, threads);
        tests::require_equal(output, expected);
        const auto resolved = resolved_thread_count(threads, 3);
        MF_REQUIRE(resolved >= 1 && resolved <= 3);
        if (threads != 0) MF_REQUIRE(resolved == std::min<std::size_t>(threads, 3));
    }
    multiply_parallel(a, b, output, std::numeric_limits<std::size_t>::max(),
                      std::numeric_limits<std::size_t>::max());
    tests::require_equal(output, expected);
    multiply_blocked(a, b, output, std::numeric_limits<std::size_t>::max());
    tests::require_equal(output, expected);
    MF_REQUIRE(resolved_thread_count(std::numeric_limits<std::size_t>::max(), 3) == 3);
    MF_REQUIRE(resolved_thread_count(0, 0) == 0);
}
