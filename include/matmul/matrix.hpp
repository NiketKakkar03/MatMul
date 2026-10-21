#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace matmul {

// Owns contiguous FP32 row-major storage. Zero dimensions are supported.
class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols, float value = 0.0F);
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    [[nodiscard]] static Matrix random(std::size_t rows, std::size_t cols,
                                       std::uint32_t seed, float low = -1, float high = 1);
    [[nodiscard]] static Matrix identity(std::size_t size);
    float& operator()(std::size_t row, std::size_t col) {
#ifndef NDEBUG
        if (row >= rows_ || col >= cols_) throw std::out_of_range("matrix index out of bounds");
#endif
        return data_[row * cols_ + col];
    }
    const float& operator()(std::size_t row, std::size_t col) const {
#ifndef NDEBUG
        if (row >= rows_ || col >= cols_) throw std::out_of_range("matrix index out of bounds");
#endif
        return data_[row * cols_ + col];
    }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] float* data() noexcept { return data_.data(); }
    [[nodiscard]] const float* data() const noexcept { return data_.data(); }
    void fill(float value);
private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> data_;
};
}
