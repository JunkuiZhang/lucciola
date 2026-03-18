#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels {

/**
 * @brief Computes Y = X * W^T
 * 
 * @param output Output tensor of shape [m, n]
 * @param input Input tensor X of shape [m, k]
 * @param weight Weight matrix W of shape [n, k]
 * @param m Number of rows in X (typically num_tokens)
 * @param n Number of rows in W, and columns in Output (out_features)
 * @param k Number of columns in X, and columns in W due to transposition (in_features)
 * @param stream CUDA stream
 */
void linear_forward(
    void *output,
    const void *input,
    const void *weight,
    const int m,
    const int n,
    const int k,
    cudaStream_t stream = 0);

} // namespace lucciola::kernels
