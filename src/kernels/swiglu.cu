#include "kernels/swiglu.h"
#include "kernels/util.h"
#include <cuda_bf16.h>
#include <math.h>

namespace lucciola::kernels {

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void swiglu_forward_kernel(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ up,
    const int num_elements) {

    // Entirely flat vectorization using 1D coordinates
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int bfloat_count = 8;
    int pack_idx = tid;
    int num_packs = num_elements / bfloat_count;

    if (pack_idx < num_packs) {
        Pack128<__nv_bfloat16> gate_pack =
            load_128bit<__nv_bfloat16>(gate + pack_idx * bfloat_count);
        Pack128<__nv_bfloat16> up_pack =
            load_128bit<__nv_bfloat16>(up + pack_idx * bfloat_count);
        Pack128<__nv_bfloat16> out_pack;

#pragma unroll
        for (int i = 0; i < bfloat_count; ++i) {
            float g = __bfloat162float(gate_pack.elements[i]);
            float u = __bfloat162float(up_pack.elements[i]);
            // out = SiLU(gate) * up
            float res = silu(g) * u;
            out_pack.elements[i] = __float2bfloat16(res);
        }

        store_128bit(out + pack_idx * bfloat_count, out_pack);
    }
}

void swiglu_forward(
    void *out,
    const void *gate,
    const void *up,
    const int num_tokens,
    const int hidden_dim,
    cudaStream_t stream) {

    // Treat [num_tokens, hidden_dim] as one completely flat 1D array
    int num_elements = num_tokens * hidden_dim;
    int num_packs = num_elements / 8;

    int threads = 256;
    int blocks = (num_packs + threads - 1) / threads;

    swiglu_forward_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        reinterpret_cast<const __nv_bfloat16 *>(gate),
        reinterpret_cast<const __nv_bfloat16 *>(up),
        num_elements);
}

} // namespace lucciola::kernels
