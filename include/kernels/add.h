#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels {

void add_forward(
    void *in_out,
    const void *addend,
    const int num_elements,
    cudaStream_t stream = nullptr);

} // namespace lucciola::kernels
