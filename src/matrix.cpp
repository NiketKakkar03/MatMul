#include "matmul/matrix.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace matmul {
namespace {
std::size_t element_count(std::size_t rows, std::size_t cols) {
    if (cols != 0 && rows > std::vector<float>().max_size() / cols)
        throw std::length_error("matrix dimensions exceed addressable storage");
    return rows * cols;
}
}
Matrix::Matrix(std::size_t rows, std::size_t cols, float value)
    : rows_(rows), cols_(cols), data_(element_count(rows, cols), value) {}
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        // Stage allocation before changing dimensions, preserving the invariant
        // even when copying a larger matrix fails with bad_alloc.
        Matrix copy(other);
        *this = std::move(copy);
    }
    return *this;
}
Matrix::Matrix(Matrix&& other) noexcept
    : rows_(std::exchange(other.rows_, 0)), cols_(std::exchange(other.cols_, 0)),
      data_(std::move(other.data_)) { other.data_.clear(); }
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        data_ = std::move(other.data_);
        other.data_.clear();
    }
    return *this;
}
Matrix Matrix::random(std::size_t rows, std::size_t cols, std::uint32_t seed,
                      float low, float high) {
    if (!std::isfinite(low) || !std::isfinite(high) || low > high)
        throw std::invalid_argument("random bounds must be finite and ordered");
    Matrix result(rows, cols);
    std::mt19937 generator(seed);
    // Explicit mapping keeps input reproducible across standard libraries.
    for (auto& value : result.data_) {
        const double unit = static_cast<double>(generator()) / 4294967296.0;
        value = static_cast<float>(static_cast<double>(low) +
                (static_cast<double>(high) - low) * unit);
    }
    return result;
}
Matrix Matrix::identity(std::size_t size) {
    Matrix result(size, size);
    for (std::size_t i = 0; i < size; ++i) result(i, i) = 1;
    return result;
}
void Matrix::fill(float value) { std::fill(data_.begin(), data_.end(), value); }
}
