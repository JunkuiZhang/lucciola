#pragma once

#include <string>

namespace lucciola {

class SafeTensors {
  public:
    SafeTensors(const std::string &model_path);
    ~SafeTensors();

  private:
};

} // namespace lucciola