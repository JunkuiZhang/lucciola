#pragma once

#include "types.h"
#include <cstddef>
#include <string>
#include <vector>

namespace lucciola {

struct TensorInfo {
    std::string name;
    TensorType type;
    std::vector<int64_t> shape;
    char *data_ptr = nullptr;
    size_t total_bytes;
};

class SafeTensors {
  public:
    SafeTensors();
    ~SafeTensors();

    bool load(const std::string filePath);

  private:
    int fd = -1;
    size_t file_size = 0;
    char *mapped = nullptr;
};

} // namespace lucciola