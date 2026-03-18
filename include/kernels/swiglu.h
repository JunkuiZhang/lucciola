#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels {

/**
 * @brief Computes SiLU(gate) * up component of SwiGLU FFN
 *
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * Output = SiLU(gate) * up
 *
 * @param out Output tensor of shape [num_tokens, hidden_dim]
 * @param gate Gate tensor of shape [num_tokens, hidden_dim]
 * @param up Up tensor of shape [num_tokens, hidden_dim]
 * @param num_tokens Number of tokens across all batches
 * @param hidden_dim Hidden dimension of the intermediate FFN
 * @param stream CUDA stream
 */
void swiglu_forward(
    void *out,
    const void *gate,
    const void *up,
    const int num_tokens,
    const int hidden_dim,
    cudaStream_t stream = 0);

} // namespace lucciola::kernels
