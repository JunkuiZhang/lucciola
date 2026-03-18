#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels {

/**
 * @brief Greedy Decoding Argmax Operator
 *
 * Takes the [num_tokens, vocab_size] logits array and finds the index of the
 * highest probability class for each token. Used at the very end of the
 * Language Model Head to determine the generated token IDs.
 *
 * @param out_tokens Output integer array of shape [num_tokens]
 * @param logits Input logits tensor of shape [num_tokens, vocab_size]
 * @param num_tokens Number of tokens (batch_size * seq_len)
 * @param vocab_size Vocabulary size (e.g., 151936 for Qwen)
 * @param stream CUDA stream
 */
void argmax_forward(
    int *out_tokens,
    const void *logits,
    const int num_tokens,
    const int vocab_size,
    cudaStream_t stream = 0);

} // namespace lucciola::kernels
