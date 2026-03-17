#include "gpu_arena.h"
#include <cuda_runtime_api.h>

namespace lucciola {

GpuArena::GpuArena() noexcept {}

GpuArena::~GpuArena() {
    if (buffer != nullptr)
        cudaFree(buffer);
    buffer_size = 0;
    current_offset = 0;
}

bool GpuArena::init(size_t size) {

    auto ret = cudaMalloc(&buffer, size);
    if (ret != cudaSuccess) {
        return false;
    }

    buffer_size = size;
    current_offset = 0;

    return true;
}

} // namespace lucciola