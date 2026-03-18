#pragma once

#include <cuda_runtime.h>

namespace lucciola::kernels {

void kv_cache_attention_forward(
    void *out,
    const void *q,
    const void *k,
    const void *v,
    void *k_cache,
    void *v_cache,
    const int seq_len,
    const int kv_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    cudaStream_t stream = nullptr);

} // namespace lucciola::kernels
