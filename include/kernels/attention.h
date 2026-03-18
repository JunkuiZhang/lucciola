#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels {

/**
 * @brief Naive Single-Batch Attention Forward (Prefill Stage)
 *
 * Divides Attention into 3 absolutely pure mathematical steps:
 * 1. Score = Q * K^T / sqrt(d)
 * 2. Prob = Softmax(Mask(Score))
 * 3. Out = Prob * V
 *
 * @param out Output tensor [seq_len, num_heads, head_dim]
 * @param q Query tensor [seq_len, num_heads, head_dim]
 * @param k Key tensor [seq_len, num_heads, head_dim]
 * @param v Value tensor [seq_len, num_heads, head_dim]
 * @param seq_len Sequence Length
 * @param num_heads Number of attention heads
 * @param head_dim Hidden dimension per head
 * @param stream CUDA stream
 */
void naive_attention_forward(
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    cudaStream_t stream = 0);

} // namespace lucciola::kernels
