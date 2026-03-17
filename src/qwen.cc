#include "qwen.h"
#include "safetensores.h"
#include <print>

namespace lucciola {

QwenModel::QwenModel(const std::string &model_path) {
    std::string safetensors_path = model_path + "/model.safetensors";
    auto safetensores = SafeTensors();
    bool load_success = safetensores.load(safetensors_path);
    if (!load_success) {
        std::println("Failed to load safetensores");
    }
}

QwenModel::~QwenModel() {}

} // namespace lucciola