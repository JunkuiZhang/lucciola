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

// void prefill_attention_forward(
//     void *out,
//     const void *q,
//     const void *k,
//     const void *v,
//     const int seq_len,
//     const int num_heads,
//     const int num_kv_heads,
//     const int head_dim,
//     cudaStream_t stream = nullptr);

// PagedAttention ??????????????????
void paged_attention_forward(
    void *out,               // [num_seqs, num_heads, head_dim]
    const void *q,           // [num_seqs, num_heads, head_dim]
    void *k_cache,           // [num_blocks, block_size, num_kv_heads, head_dim]
    void *v_cache,           // [num_blocks, block_size, num_kv_heads, head_dim]
    const int *context_lens, // [num_seqs]
    const int *block_tables, // [num_seqs, max_blocks_per_seq]
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    cudaStream_t stream = nullptr);

void append_paged_kv_cache(
    void *k_cache,
    void *v_cache,
    const void *k_new,
    const void *v_new,
    const int *context_lens,
    const int *block_tables,
    const int num_seqs,
    const int seq_len_per_req,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    cudaStream_t stream = nullptr);

void chunked_paged_attention_forward(
    void *out,     // [num_seqs, chunk_size, num_heads, head_dim]
    const void *q, // [num_seqs, chunk_size, num_heads, head_dim]
    void *k_cache,
    void *v_cache,
    const int *block_tables, // [num_seqs, max_blocks_per_seq]
    const int
        *context_lens, // [num_seqs] : total context length up to this chunk
    const int num_seqs,
    const int chunk_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    cudaStream_t stream = nullptr);

} // namespace lucciola::kernels
