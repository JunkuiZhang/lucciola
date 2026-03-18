#include "kernels/add.h"
#include <cuda_bf16.h>

namespace lucciola::kernels {

__global__ void add_kernel(
    __nv_bfloat16 *__restrict__ in_out,
    const __nv_bfloat16 *__restrict__ addend,
    const int num_elements) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        in_out[i] = __float2bfloat16(__bfloat162float(in_out[i]) + __bfloat162float(addend[i]));
    }
}

void add_forward(
    void *in_out,
    const void *addend,
    const int num_elements,
    cudaStream_t stream) {
    
    dim3 block(256);
    dim3 grid((num_elements + 255) / 256);
    
    add_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(in_out),
        reinterpret_cast<const __nv_bfloat16 *>(addend),
        num_elements);
}

} // namespace lucciola::kernels
