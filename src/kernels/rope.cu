#include "kernels/rope.h"
#include <cuda_bf16.h>
#include <math.h>

namespace lucciola::kernels {

__global__ void rope_forward_kernel(
    __nv_bfloat16 *__restrict__ tensor, // [num_tokens, num_heads, head_dim] (flattened from [batch_size, seq_len, num_heads, head_dim])
    const int *__restrict__ pos_ids,    // [num_tokens] (flattened from [batch_size, seq_len], absolute seq pos per token)
    const int num_tokens,               // Total tokens across all batches (batch_size * seq_len)
    const int num_heads,                // Number of attention heads (Q or KV)
    const int head_dim,                 // Hidden dimension per head
    const float rope_theta) {           // Base rotation frequency (e.g., 1000000.0)

    // Thread Hierarchy:
    // Block -> Handles exactly 1 Head of 1 Token
    // Thread -> Handles 1 complex pair (x, y) adjacent features

    int head_idx = blockIdx.x;
    int token_idx = blockIdx.y;
    int pair_idx = threadIdx.x; // 0 to head_dim / 2 - 1

    if (pair_idx < head_dim / 2) {
        int pos = pos_ids[token_idx];

        // 1. Calculate frequency for this specific dimension pair
        // freq = 1.0 / (theta ^ (2 * pair_idx / head_dim))
        float freq = 1.0f / powf(rope_theta, (float)(2 * pair_idx) / (float)head_dim);
        float angle = (float)pos * freq;

        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        // 2. Identify memory pointer for this exact token, head, and feature pair
        long long offset = (long long)token_idx * num_heads * head_dim +
                           (long long)head_idx * head_dim +
                           (long long)(pair_idx * 2);

        __nv_bfloat16 *ptr = tensor + offset;

        // 3. Read values and convert to float
        float x = __bfloat162float(ptr[0]);
        float y = __bfloat162float(ptr[1]);

        // 4. Apply 2D Rotation matrix
        float out_x = x * cos_val - y * sin_val;
        float out_y = x * sin_val + y * cos_val;

        // 5. Write back in-place
        ptr[0] = __float2bfloat16(out_x);
        ptr[1] = __float2bfloat16(out_y);
    }
}

void rope_forward(
    void *tensor,
    const int *pos_ids,
    const int num_tokens,
    const int num_heads,
    const int head_dim,
    const float rope_theta,
    cudaStream_t stream) {

    // Usually head_dim is 128, meaning we have 64 pairs. 
    // We launch 64 threads per block.
    int threads_per_block = head_dim / 2;
    dim3 block(threads_per_block);
    
    // Grid: [num_heads, num_tokens]
    // Easily handles completely flattened continuous batching scenarios.
    dim3 grid(num_heads, num_tokens);

    rope_forward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(tensor),
        pos_ids, num_tokens, num_heads, head_dim, rope_theta);
}

} // namespace lucciola::kernels
