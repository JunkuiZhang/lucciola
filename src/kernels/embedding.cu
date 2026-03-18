#include "kernels/embedding.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace lucciola::kernels {

template <typename T>
__global__ void embedding_forward_kernel(
    T *__restrict__ output,                // [seq_len, hidden_size]
    const int *__restrict__ input_ids,     // [seq_len]
    const T *__restrict__ embedding_table, // [vocab_size, hidden_size]
    const int seq_len,
    const int hidden_size,
    const int vocab_size,
    const int tid_end) {

    // Each thread block processes one token.
    int token_idx = blockIdx.x;
    if (token_idx >= seq_len)
        return;

    int token_id = input_ids[token_idx];

    // Simplistic out-of-bounds check (clip to 0 if out of vocabulary)
    if (token_id < 0 || token_id >= vocab_size) {
        token_id = 0;
    }

    // Base pointers for this specific token
    const float4 *weight_row = reinterpret_cast<const float4 *>(
        embedding_table + token_id * hidden_size);
    float4 *output_row =
        reinterpret_cast<float4 *>(output + token_idx * hidden_size);

    // Threads cooperatively copy the embedding row to the output.
    for (int i = threadIdx.x; i < tid_end; i += blockDim.x) {
        output_row[i] = weight_row[i];
    }
}

template <>
void embedding_forward<__nv_bfloat16>(
    void *output,
    const int *input_ids,
    const void *weight,
    const int seq_len,
    const int hidden_size,
    const int vocab_size,
    cudaStream_t stream) {
    // We launch one block per sequence element.
    dim3 grid(seq_len);
    // 256 or 512 threads per block is generally good for memory copies.
    // If hidden_size is exactly 1024 or similar, 256 is enough to do 4 loops.
    int thread_count = std::min(hidden_size / 8, 256);
    dim3 block(thread_count);

    embedding_forward_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(output),
        input_ids,
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        seq_len,
        hidden_size,
        vocab_size,
        hidden_size / 8);
}

template <>
void embedding_forward<short>(
    void *output,
    const int *input_ids,
    const void *weight,
    const int seq_len,
    const int hidden_size,
    const int vocab_size,
    cudaStream_t stream) {
    embedding_forward<__nv_bfloat16>(
        output, input_ids, weight, seq_len, hidden_size, vocab_size, stream);
}

} // namespace lucciola::kernels
