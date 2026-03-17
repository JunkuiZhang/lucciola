#pragma once

#include <string>

namespace lucciola {

class SafeTensors {
  public:
    SafeTensors();
    ~SafeTensors();

    bool load(const std::string filePath);

  private:
    int fd = -1;
};

} // namespace lucciola