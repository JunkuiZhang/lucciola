#pragma once

#include <cuda_runtime.h>

namespace lucciola {
namespace kernels {

// Basic terminology:
// output: pre-allocated GPU buffer to hold the result [seq_len, hidden_size]
// input_ids: GPU buffer holding the token IDs [seq_len]
// weight: GPU buffer of the model's vocabulary embeddings [vocab_size,
// hidden_size] Only BF16 is fully specialized initially since Qwen3.0 is bf16.
template <typename T>
void embedding_forward(
    void *output,
    const int *input_ids,
    const void *weight,
    const int seq_len,
    const int hidden_size,
    const int vocab_size,
    cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace lucciola
