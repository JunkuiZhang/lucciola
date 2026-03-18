#include "kernels/linear.h"
#include "kernels/util.h"
#include <cuda_bf16.h>

namespace lucciola::kernels {

__global__ void linear_forward_kernel(
    __nv_bfloat16 *__restrict__ output, // [num_tokens, n] (flattened from
                                        // [batch_size, seq_len, out_features])
    const __nv_bfloat16
        *__restrict__ input, // [num_tokens, k] (flattened from [batch_size,
                             // seq_len, in_features])
    const __nv_bfloat16 *__restrict__ weight, // [n, k] (weight matrix
                                              // [out_features, in_features])
    const int m,   // Total tokens across all batches (batch_size * seq_len)
    const int n,   // Output features dimension (out_features)
    const int k) { // Input features dimension (in_features)

    // Thread mapped to calculate one single output element: output[row, col]
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m dimension (tokens)
    int col =
        blockIdx.x * blockDim.x + threadIdx.x; // n dimension (out_features)

    if (row < m && col < n) {
        float sum = 0.0f;
        const __nv_bfloat16 *x_row = input + row * k;
        const __nv_bfloat16 *w_row = weight + col * k;

        constexpr int bfloat_count = 4;
        int num_f2 = k / bfloat_count;

        // Vectorized dot product calculation using 8-byte float2
        for (int i = 0; i < num_f2; ++i) {
            Pack64<__nv_bfloat16> x_pack =
                load_64bit<__nv_bfloat16>(x_row + i * bfloat_count);
            Pack64<__nv_bfloat16> w_pack =
                load_64bit<__nv_bfloat16>(w_row + i * bfloat_count);

#pragma unroll
            for (int j = 0; j < bfloat_count; ++j) {
                float vx = __bfloat162float(x_pack.elements[j]);
                float vw = __bfloat162float(w_pack.elements[j]);
                sum += vx * vw;
            }
        }

        // Handle remainder if k is not a multiple of 4
        for (int i = num_f2 * bfloat_count; i < k; ++i) {
            float vx = __bfloat162float(x_row[i]);
            float vw = __bfloat162float(w_row[i]);
            sum += vx * vw;
        }

        output[row * n + col] = __float2bfloat16(sum);
    }
}

void linear_forward(
    void *output,
    const void *input,
    const void *weight,
    const int m,
    const int n,
    const int k,
    cudaStream_t stream) {

    // 2D grid for matrix multiplication element-wise coverage
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    linear_forward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(output),
        reinterpret_cast<const __nv_bfloat16 *>(input),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        m,
        n,
        k);
}

} // namespace lucciola::kernels
