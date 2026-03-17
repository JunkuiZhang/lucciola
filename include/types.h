#pragma once

#include <expected>
#include <string>

namespace lucciola {

enum TensorType {
    BF16,
};

inline std::expected<TensorType, std::string>
parse_tensor_type(std::string_view type_string) {
    if (type_string == "bf16" || type_string == "BF16") {
        return BF16;
    }
    return std::unexpected("Unknown tensor type: " + std::string(type_string));
}

} // namespace lucciola