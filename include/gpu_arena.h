#pragma once

#include <cstddef>

namespace lucciola {

class GpuArena {
  public:
    GpuArena() noexcept;
    ~GpuArena();

    bool init(size_t size);

    template <typename T> T *alloc(size_t num_elements) {
        size_t required_size = num_elements * sizeof(T);
        auto aligned_offset = (current_offset + 255) & (~255);

        if (buffer_size - aligned_offset < required_size)
            return nullptr;

        T *ptr = reinterpret_cast<T *>(static_cast<char *>(buffer) + aligned_offset);
        current_offset = aligned_offset + required_size;

        return ptr;
    }

  private:
    void *buffer = nullptr;
    size_t buffer_size = 0;

    size_t current_offset = 0;
};

} // namespace lucciola