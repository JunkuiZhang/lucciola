#include "kernels/rmsnorm.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace lucciola {
namespace kernels {

// Block-level reduction max size assumption for basic implementation.
// Assuming block size of 256.
__inline__ __device__ float block_reduce_sum(float val) {
    // Shared memory for warp reduction
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // First thread of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Now reduce the warp sums in the first warp
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

template <typename T>
__global__ void rmsnorm_forward_kernel(
    T *__restrict__ output,       // [seq_len, hidden_size]
    const T *__restrict__ input,  // [seq_len, hidden_size]
    const T *__restrict__ weight, // [hidden_size]
    const int hidden_size,
    const float eps) {

    // Each block processes one row (one token)
    int row_idx = blockIdx.x;

    const T *input_row = input + row_idx * hidden_size;
    T *output_row = output + row_idx * hidden_size;

    // 1. Calculate sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        // Convert to float for higher precision reduction
        float val = __bfloat162float(input_row[i]);
        sum_sq += val * val;
    }

    // Reduce across block
    sum_sq = block_reduce_sum(sum_sq);

    // 2. Compute root mean square
    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads(); // Wait for RMS to be computed

    // 3. Normalize and scale
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(input_row[i]);
        float w = __bfloat162float(weight[i]);
        // out = (x / rms) * weight == (x * rsqrt) * weight
        output_row[i] = __float2bfloat16(val * rms * w);
    }
}

template <>
void rmsnorm_forward<__nv_bfloat16>(
    void *output,
    const void *input,
    const void *weight,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream) {
    dim3 grid(seq_len);
    dim3 block(256); // Assuming standard block size

    rmsnorm_forward_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(output),
        reinterpret_cast<const __nv_bfloat16 *>(input),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        hidden_size,
        eps);
}

} // namespace kernels
} // namespace lucciola
