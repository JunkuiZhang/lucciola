#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels::learn {

void naive_softmax_forward(float *scores, int valid_len, cudaStream_t stream);

void online_softmax_forward(float *scores, int valid_len, cudaStream_t stream);

void warp_softmax_forward(float *scores, int valid_len, cudaStream_t stream);

void batched_warp_softmax_forward(
    float *scores, int valid_len, int num_rows, cudaStream_t stream);

} // namespace lucciola::kernels::learn
