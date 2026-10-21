#include "matmul/algorithms.hpp"
#include <stdexcept>

namespace matmul {

void validate_multiplication(const Matrix& a, const Matrix& b, const Matrix& c) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Multiplication requires A columns to equal B rows");
    }
    if (c.rows() != a.rows() || c.cols() != b.cols()) {
        throw std::invalid_argument("Output shape must equal A rows by B columns");
    }
    // Matrix owns its storage, so identity is sufficient to reject all possible
    // output/input aliases, including zero-sized matrices with null data().
    if (&c == &a || &c == &b) {
        throw std::invalid_argument("Output matrix must not alias either input");
    }
}

std::vector<std::string> implementation_names() {
    std::vector<std::string> names{
        "naive", "reordered", "ijk", "ikj", "jik", "jki", "kij", "kji", "blocked", "parallel"};
    if (accelerate_available()) names.emplace_back("accelerate");
    return names;
}

void multiply_into(const Matrix& a, const Matrix& b, Matrix& c,
                   std::string_view implementation, MultiplyOptions options) {
    if (implementation == "naive") {
        multiply_naive(a, b, c);
    } else if (implementation == "reordered") {
        multiply_reordered(a, b, c);
    } else if (implementation == "blocked") {
        multiply_blocked(a, b, c, options.block_size);
    } else if (implementation == "parallel" || implementation == "blocked_parallel") {
        multiply_parallel(a, b, c, options.block_size, options.threads);
    } else if (implementation == "accelerate") {
        multiply_accelerate(a, b, c);
    } else if (implementation == "ijk" || implementation == "ikj" || implementation == "jik" ||
               implementation == "jki" || implementation == "kij" || implementation == "kji" ||
               implementation == "i-j-k" || implementation == "i-k-j" || implementation == "j-i-k" ||
               implementation == "j-k-i" || implementation == "k-i-j" || implementation == "k-j-i") {
        multiply_loop_order(a, b, c, implementation);
    } else {
        throw std::invalid_argument("Unknown implementation: " + std::string(implementation));
    }
}

Matrix multiply(const Matrix& a, const Matrix& b, std::string_view implementation,
                MultiplyOptions options) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Multiplication requires A columns to equal B rows");
    }
    Matrix result(a.rows(), b.cols());
    multiply_into(a, b, result, implementation, options);
    return result;
}

} // namespace matmul
