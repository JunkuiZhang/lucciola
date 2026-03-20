#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels::learn {

void sgemm_naive_forward(
    float *C,
    const float *A,
    const float *B,
    int M,
    int N,
    int K,
    cudaStream_t stream);

void sgemm_tiled_forward(
    float *C,
    const float *A,
    const float *B,
    int M,
    int N,
    int K,
    cudaStream_t stream);

} // namespace lucciola::kernels::learn