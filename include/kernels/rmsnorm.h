#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels {

void rmsnorm_forward(
    void *output,
    const void *input,
    const void *weight,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream = nullptr);

} // namespace lucciola::kernels
