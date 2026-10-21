#include "test_support.hpp"
#include <limits>
#include <utility>

using matmul::Matrix;

MF_TEST(matrix_storage_and_fill) {
    Matrix matrix(2, 3, 7.0F);
    MF_REQUIRE(matrix.rows() == 2);
    MF_REQUIRE(matrix.cols() == 3);
    MF_REQUIRE(matrix.size() == 6);
    matrix(1, 2) = -4.0F;
    MF_REQUIRE(matrix.data()[5] == -4.0F);
    const Matrix& read = matrix;
    MF_REQUIRE(read(1, 2) == -4.0F);
    matrix.fill(2.0F);
    for (std::size_t i = 0; i < matrix.size(); ++i) MF_REQUIRE(matrix.data()[i] == 2.0F);
    Matrix empty(0, 7);
    MF_REQUIRE(empty.size() == 0);
    MF_REQUIRE(empty.cols() == 7);
    MF_REQUIRE(Matrix(7, 0).size() == 0);
}

MF_TEST(matrix_identity_and_random) {
    const auto identity = Matrix::identity(5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            MF_REQUIRE(identity(i, j) == (i == j ? 1.0F : 0.0F));
    MF_REQUIRE(Matrix::identity(0).size() == 0);
    const auto a = Matrix::random(9, 7, 42, -3.0F, 2.0F);
    const auto b = Matrix::random(9, 7, 42, -3.0F, 2.0F);
    bool different_seed_changes_values = false;
    const auto c = Matrix::random(9, 7, 43, -3.0F, 2.0F);
    for (std::size_t i = 0; i < a.size(); ++i) {
        MF_REQUIRE(a.data()[i] == b.data()[i]);
        MF_REQUIRE(a.data()[i] >= -3.0F && a.data()[i] <= 2.0F);
        different_seed_changes_values |= a.data()[i] != c.data()[i];
    }
    MF_REQUIRE(different_seed_changes_values);
    const auto constant = Matrix::random(2, 2, 42, 3.0F, 3.0F);
    MF_REQUIRE(constant(1, 1) == 3.0F);
    MF_THROWS(std::invalid_argument, Matrix::random(1, 1, 1, 2.0F, -2.0F));
    MF_THROWS(std::invalid_argument,
              Matrix::random(1, 1, 1, std::numeric_limits<float>::quiet_NaN(), 1.0F));
    MF_THROWS(std::invalid_argument,
              Matrix::random(1, 1, 1, 0.0F, std::numeric_limits<float>::infinity()));
}

MF_TEST(matrix_copy_and_move_ownership) {
    Matrix original = Matrix::random(3, 4, 17);
    Matrix copy = original;
    tests::require_equal(copy, original, 0);
    copy(0, 0) = 99;
    MF_REQUIRE(original(0, 0) != 99);
    Matrix assigned(1, 1);
    assigned = original;
    tests::require_equal(assigned, original, 0);
    const Matrix& self = assigned;
    assigned = self;
    tests::require_equal(assigned, original, 0);
    const auto* storage = assigned.data();
    Matrix moved(std::move(assigned));
    MF_REQUIRE(moved.data() == storage);
    tests::require_equal(moved, original, 0);
    MF_REQUIRE(assigned.rows() == 0 && assigned.cols() == 0 && assigned.size() == 0);
    Matrix destination(2, 2);
    destination = std::move(moved);
    MF_REQUIRE(destination.data() == storage);
    tests::require_equal(destination, original, 0);
    MF_REQUIRE(moved.rows() == 0 && moved.cols() == 0 && moved.size() == 0);
    assigned = Matrix(1, 2, 4);
    MF_REQUIRE(assigned(0, 1) == 4);
}

MF_TEST(matrix_overflow_and_debug_bounds) {
    MF_THROWS(std::length_error, Matrix(std::numeric_limits<std::size_t>::max(), 2));
    MF_THROWS(std::length_error,
              Matrix(std::numeric_limits<std::size_t>::max() / sizeof(float) + 1, 1));
#ifndef NDEBUG
    Matrix matrix(2, 3);
    MF_THROWS(std::out_of_range, matrix(2, 0));
    MF_THROWS(std::out_of_range, matrix(0, 3));
    const Matrix& immutable = matrix;
    MF_THROWS(std::out_of_range, immutable(2, 0));
    Matrix empty(0, 0);
    MF_THROWS(std::out_of_range, empty(0, 0));
#endif
}

int main() {
    std::size_t failures = 0;
    for (const auto& [name, test] : tests::registry()) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": unknown exception\n";
        }
    }
    std::cout << tests::registry().size() - failures << '/' << tests::registry().size()
              << " tests passed\n";
    return failures == 0 ? 0 : 1;
}
