#include "test_support.hpp"
#include <limits>

using namespace matmul;

MF_TEST(empty_multiplication_shapes) {
    for (const auto& name : implementation_names()) {
        const auto empty_rows = multiply(Matrix(0, 3), Matrix(3, 4), name);
        MF_REQUIRE(empty_rows.rows() == 0 && empty_rows.cols() == 4);
        const auto empty_columns = multiply(Matrix(2, 3), Matrix(3, 0), name);
        MF_REQUIRE(empty_columns.rows() == 2 && empty_columns.cols() == 0);
        Matrix output(3, 4, 27);
        multiply_into(Matrix(3, 0), Matrix(0, 4), output, name);
        tests::require_equal(output, Matrix(3, 4), 0);
    }
}

MF_TEST(empty_products_with_maximum_dimension) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    const Matrix tall_empty(maximum, 0);
    const Matrix wide_empty(0, maximum);
    const Matrix empty(0, 0);
    const auto tall_reference = reference_multiply(tall_empty, empty);
    MF_REQUIRE(tall_reference.rows() == maximum && tall_reference.cols() == 0);
    const auto wide_reference = reference_multiply(empty, wide_empty);
    MF_REQUIRE(wide_reference.rows() == 0 && wide_reference.cols() == maximum);
    for (const auto& name : implementation_names()) {
        const auto tall_output = multiply(tall_empty, empty, name);
        MF_REQUIRE(tall_output.rows() == maximum && tall_output.cols() == 0);
        const auto wide_output = multiply(empty, wide_empty, name);
        MF_REQUIRE(wide_output.rows() == 0 && wide_output.cols() == maximum);
    }
}

MF_TEST(invalid_dimensions_aliases_and_options) {
    Matrix a(2, 3), b(3, 4), good(2, 4), bad(4, 2);
    MF_THROWS(std::invalid_argument, validate_multiplication(a, b, bad));
    MF_THROWS(std::invalid_argument, multiply(a, Matrix(2, 4)));
    for (const auto& name : implementation_names()) {
        MF_THROWS(std::invalid_argument, multiply_into(a, b, bad, name));
        MF_THROWS(std::invalid_argument, multiply_into(a, Matrix(2, 4), good, name));
        Matrix square(3, 3);
        MF_THROWS(std::invalid_argument, multiply_into(square, square, square, name));
        Matrix left(3, 3), right(3, 3);
        MF_THROWS(std::invalid_argument, multiply_into(left, right, left, name));
        MF_THROWS(std::invalid_argument, multiply_into(left, right, right, name));
        Matrix empty(0, 0);
        MF_THROWS(std::invalid_argument, multiply_into(empty, empty, empty, name));
    }
    MF_THROWS(std::invalid_argument, multiply_into(a, b, good, "missing"));
    MF_THROWS(std::invalid_argument, multiply_loop_order(a, b, good, "iij"));
    MF_THROWS(std::invalid_argument, multiply_loop_order(a, b, good, ""));
    MF_THROWS(std::invalid_argument, multiply_blocked(a, b, good, 0));
    MF_THROWS(std::invalid_argument, multiply_parallel(a, b, good, 0, 2));
}

MF_TEST(comparison_tolerance_and_nonfinite_rejection) {
    Matrix a(1, 2), b(1, 2);
    a(0, 0) = 1.0F; b(0, 0) = 1.00001F;
    MF_REQUIRE(compare(a, b).correct);
    MF_REQUIRE(!compare(a, b, 0, 0).correct);
    MF_REQUIRE(compare(a, b).maximum_absolute_error > 0);
    MF_REQUIRE(compare(a, b).maximum_relative_error > 0);
    MF_REQUIRE(compare(a, b).maximum_relative_error == compare(b, a).maximum_relative_error);
    a(0, 0) = 0; b(0, 0) = 0.00005F;
    MF_REQUIRE(compare(a, b, 0.0001, 0).correct);
    a(0, 0) = 1000; b(0, 0) = 1001;
    MF_REQUIRE(compare(a, b, 0, 0.002).correct);
    for (const auto value : {std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
        a(0, 0) = value; b(0, 0) = value;
        MF_REQUIRE(!compare(a, b).correct);
        MF_REQUIRE(!approximately_equal(a, b));
        b(0, 0) = 1;
        MF_REQUIRE(!compare(a, b).correct);
        MF_REQUIRE(!compare(b, a).correct);
    }
    MF_THROWS(std::invalid_argument, compare(a, Matrix(2, 1)));
    MF_THROWS(std::invalid_argument, compare(a, b, -1, 0));
    MF_THROWS(std::invalid_argument, compare(a, b, 0, -1));
    MF_THROWS(std::invalid_argument,
              compare(a, b, std::numeric_limits<double>::quiet_NaN(), 0));
    MF_THROWS(std::invalid_argument,
              compare(a, b, 0, std::numeric_limits<double>::infinity()));
}
