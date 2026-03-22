#pragma once

namespace lucciola::kernels {

// 宿主端调用包装器
void launch_lucciola_attention(
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const int *block_tables,
    const int *context_lens,
    int num_seqs,
    int q_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    cudaStream_t stream);

} // namespace lucciola::kernels