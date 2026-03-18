#pragma once

#include "types.h"
#include <cstddef>
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

    const void *get_tensor(const std::string &name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return it->second.data_ptr;
        }
        printf("CRITICAL ERROR: Tensor not found in model file: %s\n", name.c_str());
        return nullptr;
    }

  private:
    int fd = -1;
    size_t file_size = 0;
    char *mapped = nullptr;

    std::unordered_map<std::string, TensorInfo> tensors;
};

} // namespace lucciola