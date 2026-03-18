#pragma once

#include "tensor.h"
#include "types.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace lucciola {

struct TensorInfo {
    TensorType type;
    std::vector<int64_t> shape;
    char *data_ptr = nullptr;
    size_t total_bytes;
};

class SafeTensors {
  public:
    SafeTensors();
    ~SafeTensors();

    bool load(const std::string &file_path);

    template <typename T> Tensor<T> get_tensor(const std::string &name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            T *device_ptr = nullptr;
            cudaMalloc(&device_ptr, it->second.total_bytes);
            cudaMemcpy(
                device_ptr,
                it->second.data_ptr,
                it->second.total_bytes,
                cudaMemcpyDefault);
            return Tensor<T>(device_ptr, it->second.type, it->second.shape);
        }
        printf(
            "CRITICAL ERROR: Tensor not found in model file: %s\n",
            name.c_str());
        return Tensor<T>(nullptr, TensorType::BF16, {});
    }

  private:
    int fd = -1;
    size_t file_size = 0;
    char *mapped = nullptr;

    std::unordered_map<std::string, TensorInfo> tensors;
};

} // namespace lucciola