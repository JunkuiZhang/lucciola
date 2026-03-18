#include "kernels/rmsnorm.h"
#include "kernels/util.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define NUM_THREADS 256

namespace lucciola::kernels {

__forceinline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction max size assumption for basic implementation.
// Assuming block size of 256.
__forceinline__ __device__ float block_reduce_sum(float val) {
    // Shared memory for warp reduction
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp level reduction using shuffle
    val = warp_reduce_sum(val);

    // First thread of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Now reduce the warp sums in the first warp
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void rmsnorm_forward_kernel(
    __nv_bfloat16 *__restrict__ output,       // [seq_len, hidden_size]
    const __nv_bfloat16 *__restrict__ input,  // [seq_len, hidden_size]
    const __nv_bfloat16 *__restrict__ weight, // [hidden_size]
    const int hidden_size,
    const float eps) {

    // Each block processes one row (one token)
    int row_idx = blockIdx.x;

    const __nv_bfloat16 *input_row = input + row_idx * hidden_size;
    __nv_bfloat16 *output_row = output + row_idx * hidden_size;
    constexpr int bfloat_count = 4;

    // 1. Calculate sum of squares
    float sum_sq = 0.0f;
    int num_f4 = hidden_size / bfloat_count;

    for (int i = threadIdx.x; i < num_f4; i += blockDim.x) {
        Pack64<__nv_bfloat16> pack =
            load_64bit<__nv_bfloat16>(input_row + i * bfloat_count);

#pragma unroll
        for (int j = 0; j < bfloat_count; ++j) {
            float val = __bfloat162float(pack.elements[j]);
            sum_sq += val * val;
        }
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
    for (int i = threadIdx.x; i < num_f4; i += blockDim.x) {
        Pack64<__nv_bfloat16> in_pack =
            load_64bit<__nv_bfloat16>(input_row + i * bfloat_count);
        Pack64<__nv_bfloat16> w_pack =
            load_64bit<__nv_bfloat16>(weight + i * bfloat_count);
        Pack64<__nv_bfloat16> out_pack;

#pragma unroll
        for (int j = 0; j < bfloat_count; ++j) {
            float val = __bfloat162float(in_pack.elements[j]);
            float w = __bfloat162float(w_pack.elements[j]);
            // out = (x / rms) * weight == (x * rsqrt) * weight
            out_pack.elements[j] = __float2bfloat16(val * rms * w);
        }
        store_64bit(output_row + i * bfloat_count, out_pack);
    }
}

void rmsnorm_forward(
    void *output,
    const void *input,
    const void *weight,
    const int seq_len,
    const int hidden_size,
    const float eps,
    cudaStream_t stream) {
    dim3 grid(seq_len);
    dim3 block(NUM_THREADS);

    rmsnorm_forward_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(output),
        reinterpret_cast<const __nv_bfloat16 *>(input),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        hidden_size,
        eps);
}

} // namespace lucciola::kernels
