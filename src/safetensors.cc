#include "safetensores.h"
#include <cstdint>
#include <fcntl.h>
#include <print>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

namespace lucciola {

SafeTensors::SafeTensors() {}

SafeTensors::~SafeTensors() {}

bool SafeTensors::load(const std::string filePath) {
    fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1)
        return false;

    struct stat sb;
    if (fstat(fd, &sb) == -1)
        return false;

    auto file_size = sb.st_size;

    // mmap
    auto mapped = static_cast<char *>(
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
    std::println("{}", header_json.dump());

    return true;
}

} // namespace lucciola