#include "kernels/argmax.h"
#include <float.h>

#include <cuda_bf16.h>

namespace lucciola::kernels {

__global__ void argmax_kernel(
    int *__restrict__ out_tokens,
    const __nv_bfloat16 *__restrict__ logits,
    const int vocab_size) {

    // One block handles exactly one token's entire vocabulary distribution
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Pointer to the logits for this specific token
    const __nv_bfloat16 *token_logits = logits + token_idx * vocab_size;

    // 1. Thread-local max finding
    // Every thread strides through the vocabulary and finds its own local
    // maximum
    float local_max_val = -FLT_MAX;
    int local_max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(token_logits[i]);
        if (val > local_max_val) {
            local_max_val = val;
            local_max_idx = i;
        }
    }

    // 2. Block reduction using Shared Memory
    // We must communicate across the block to find the absolute global maximum.
    __shared__ float shared_max_vals[256];
    __shared__ int shared_max_idxs[256];

    shared_max_vals[tid] = local_max_val;
    shared_max_idxs[tid] = local_max_idx;

    __syncthreads();

    // Standard Tree Reduction algorithm (O(logN))
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max_vals[tid + stride] > shared_max_vals[tid]) {
                shared_max_vals[tid] = shared_max_vals[tid + stride];
                shared_max_idxs[tid] = shared_max_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // 3. Thread 0 writes the final winning token index
    if (tid == 0) {
        out_tokens[token_idx] = shared_max_idxs[0];
    }
}

void argmax_forward(
    int *out_tokens,
    const void *logits,
    const int num_tokens,
    const int vocab_size,
    cudaStream_t stream) {

    // Grid: exactly one block per sequence token
    dim3 grid(num_tokens);
    // Block: 256 parallel workers to divide and conquer the vocabulary words
    dim3 block(256);

    // Note: Assuming block size is strictly 256 for the static shared memory
    // allocation
    argmax_kernel<<<grid, block, 0, stream>>>(
        out_tokens,
        reinterpret_cast<const __nv_bfloat16 *>(logits),
        vocab_size);
}

} // namespace lucciola::kernels
