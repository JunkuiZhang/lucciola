#include "safetensores.h"
#include <cstdint>
#include <fcntl.h>
#include <print>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

namespace lucciola {

SafeTensors::SafeTensors() {}

SafeTensors::~SafeTensors() {
    if (mapped != nullptr)
        cudaHostUnregister(mapped);
    if (mapped != MAP_FAILED && mapped != nullptr)
        munmap(mapped, file_size);
    if (fd != -1)
        close(fd);
}

bool SafeTensors::load(const std::string &file_path) {
    fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1)
        return false;

    struct stat sb;
    if (fstat(fd, &sb) == -1)
        return false;

    file_size = sb.st_size;

    // mmap
    mapped = static_cast<char *>(
        mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (mapped == MAP_FAILED)
        return false;

    // Pinned memory
    auto pinned_ret =
        cudaHostRegister(mapped, file_size, cudaHostRegisterReadOnly);
    if (pinned_ret != cudaSuccess)
        return false;

    uint64_t header_size = *reinterpret_cast<uint64_t *>(mapped);
    std::string header_string(mapped + 8, header_size);
    nlohmann::json header_json = nlohmann::json::parse(header_string);

    auto tensors_data_base_offset = 8 + header_size;

    for (auto &tensor : header_json.items()) {
        if (tensor.key() == "__metadata__") {
            continue;
        }

        TensorInfo tensor_info;
        auto tensor_json = tensor.value();
        auto name = tensor.key();

        auto type_expected =
            parse_tensor_type(tensor_json["dtype"].get<std::string>());
        if (!type_expected) {
            std::println(stderr, "{}", type_expected.error());
            return false;
        }

        tensor_info.type = type_expected.value();

        tensor_info.shape = tensor_json["shape"].get<std::vector<int64_t>>();

        size_t start_offset = tensor_json["data_offsets"][0].get<size_t>();
        size_t end_offset = tensor_json["data_offsets"][1].get<size_t>();

        tensor_info.total_bytes = end_offset - start_offset;
        tensor_info.data_ptr = mapped + tensors_data_base_offset + start_offset;

        // std::println(
        //     "{} {} {}", name, tensor_info.shape, tensor_info.total_bytes);

        tensors[name] = tensor_info;
    }

    return true;
}

} // namespace lucciola