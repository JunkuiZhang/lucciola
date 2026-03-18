#pragma once

#include "types.h"
#include <cuda_runtime.h>
#include <vector>

namespace lucciola {

template <typename T> class Tensor {
  public:
    Tensor() = default;
    Tensor(T *device_ptr, TensorType type, std::vector<int64_t> shape)
        : device_ptr_(device_ptr), type_(type), shape_(shape) {}

    ~Tensor() {
        if (device_ptr_ != nullptr) {
            cudaFree(device_ptr_);
        }
    }

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&other) noexcept
        : device_ptr_(other.device_ptr_), type_(other.type_),
          shape_(std::move(other.shape_)) {
        other.device_ptr_ = nullptr;
    }

    Tensor &operator=(Tensor &&other) noexcept {
        if (this != &other) {
            if (device_ptr_ != nullptr) {
                cudaFree(device_ptr_);
            }
            device_ptr_ = other.device_ptr_;
            type_ = other.type_;
            shape_ = std::move(other.shape_);
            other.device_ptr_ = nullptr;
        }
        return *this;
    }

    T *data() const { return device_ptr_; }
    operator T *() const { return device_ptr_; }
    TensorType type() const { return type_; }
    const std::vector<int64_t> &shape() const { return shape_; }

  private:
    T *device_ptr_;
    TensorType type_;
    std::vector<int64_t> shape_;
};

} // namespace lucciola
