#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels {

/**
 * @brief Applies Rotary Position Embedding to a tensor in-place.
 * 
 * This operator is naturally compatible with continuous batching (flattening)
 * because the explicit `pos_ids` array provides the absolute sequence position
 * mapping for every individual token.
 * 
 * @param tensor Tensor of shape [num_tokens, num_heads, head_dim] (in-place modification)
 * @param pos_ids Array of size [num_tokens] representing absolute sequence positions
 * @param num_tokens Number of tokens across all batches
 * @param num_heads Number of attention heads (Q or KV)
 * @param head_dim Hidden dimension per head
 * @param rope_theta The base rotation frequency parameter (e.g., 1000000.0)
 * @param stream CUDA stream
 */
void rope_forward(
    void *tensor,
    const int *pos_ids,
    const int num_tokens,
    const int num_heads,
    const int head_dim,
    const float rope_theta,
    cudaStream_t stream = 0);

} // namespace lucciola::kernels
