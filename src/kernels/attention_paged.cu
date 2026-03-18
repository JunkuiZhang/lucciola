#include "kernels/attention.h"
#include <cuda_bf16.h>
#include <math.h>

namespace lucciola::kernels {

__global__ void append_paged_kv_cache_kernel(
    __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache,
    const __nv_bfloat16 *__restrict__ k_new,
    const __nv_bfloat16 *__restrict__ v_new,
    const int *__restrict__ context_lens,
    const int *__restrict__ block_tables,
    const int num_seqs,
    const int seq_len,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq) {

    int i = blockIdx.x;
    int s = blockIdx.y;
    int h = threadIdx.y;
    int d = threadIdx.x;

    if (s < num_seqs && i < seq_len && h < num_kv_heads && d < head_dim) {
        int pos = context_lens[s] - seq_len + i;
        int logical_block_idx = pos / block_size;
        int physical_block_idx =
            block_tables[s * max_blocks_per_seq + logical_block_idx];
        int block_offset = pos % block_size;

        long long dst_idx = (long long)physical_block_idx * block_size *
                                num_kv_heads * head_dim +
                            (long long)block_offset * num_kv_heads * head_dim +
                            (long long)h * head_dim + d;

        long long src_idx = (long long)s * seq_len * num_kv_heads * head_dim +
                            (long long)i * num_kv_heads * head_dim +
                            (long long)h * head_dim + d;

        k_cache[dst_idx] = k_new[src_idx];
        v_cache[dst_idx] = v_new[src_idx];
    }
}

__global__ void paged_attention_kernel(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const __nv_bfloat16 *__restrict__ v_cache,
    const int *__restrict__ context_lens,
    const int *__restrict__ block_tables,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq) {

    int s = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;

    if (s >= num_seqs || h >= num_heads)
        return;

    int kv_h = h / (num_heads / num_kv_heads);
    int prompt_len = context_lens[s];

    const __nv_bfloat16 *q_row = q + s * num_heads * head_dim + h * head_dim;

    extern __shared__ float q_smem[];
    if (tid < head_dim) {
        q_smem[tid] = __bfloat162float(q_row[tid]);
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)head_dim);
    __shared__ float scores[1024];
    if (tid < prompt_len && tid < 1024) {
        int logical_block_idx = tid / block_size;
        int physical_block_idx =
            block_tables[s * max_blocks_per_seq + logical_block_idx];
        int block_offset = tid % block_size;

        long long k_idx = (long long)physical_block_idx * block_size *
                              num_kv_heads * head_dim +
                          (long long)block_offset * num_kv_heads * head_dim +
                          (long long)kv_h * head_dim;

        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sum += q_smem[d] * __bfloat162float(k_cache[k_idx + d]);
        }
        scores[tid] = sum * scale;
    } else if (tid < 1024) {
        scores[tid] = -1e20f;
    }
    __syncthreads();

    float local_max = -1e20f;
    for (int i = 0; i < prompt_len && i < 1024; i++) {
        local_max = fmaxf(local_max, scores[i]);
    }

    if (tid < prompt_len && tid < 1024) {
        scores[tid] = expf(scores[tid] - local_max);
    }
    __syncthreads();

    if (tid == 0) {
        float sum_exp = 0.0f;
        for (int i = 0; i < prompt_len && i < 1024; i++)
            sum_exp += scores[i];
        for (int i = 0; i < prompt_len && i < 1024; i++)
            scores[i] /= sum_exp;
    }
    __syncthreads();

    if (tid < head_dim) {
        float out_val = 0.0f;
        for (int j = 0; j < prompt_len && j < 1024; ++j) {
            int logical_block_idx = j / block_size;
            int physical_block_idx =
                block_tables[s * max_blocks_per_seq + logical_block_idx];
            int block_offset = j % block_size;

            long long v_idx =
                (long long)physical_block_idx * block_size * num_kv_heads *
                    head_dim +
                (long long)block_offset * num_kv_heads * head_dim +
                (long long)kv_h * head_dim + tid;
            out_val += scores[j] * __bfloat162float(v_cache[v_idx]);
        }

        long long out_idx =
            (long long)s * num_heads * head_dim + (long long)h * head_dim + tid;
        out[out_idx] = __float2bfloat16(out_val);
    }
}

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
    cudaStream_t stream) {

    dim3 grid(seq_len_per_req, num_seqs);
    dim3 block(head_dim, num_kv_heads);

    if (head_dim * num_kv_heads > 1024) {
        block = dim3(16, 16);
        grid = dim3(
            (head_dim + 15) / 16,
            (num_kv_heads + 15) / 16,
            num_seqs * seq_len_per_req);
    }

    append_paged_kv_cache_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(k_cache),
        reinterpret_cast<__nv_bfloat16 *>(v_cache),
        reinterpret_cast<const __nv_bfloat16 *>(k_new),
        reinterpret_cast<const __nv_bfloat16 *>(v_new),
        context_lens,
        block_tables,
        num_seqs,
        seq_len_per_req,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq);
}

void paged_attention_forward(
    void *out,
    const void *q,
    void *k_cache,
    void *v_cache,
    const int *context_lens,
    const int *block_tables,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    cudaStream_t stream) {

    dim3 grid(num_heads, num_seqs);
    int threads = 1024;
    size_t smem = head_dim * sizeof(float);

    paged_attention_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        reinterpret_cast<const __nv_bfloat16 *>(q),
        reinterpret_cast<const __nv_bfloat16 *>(k_cache),
        reinterpret_cast<const __nv_bfloat16 *>(v_cache),
        context_lens,
        block_tables,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq);
}

} // namespace lucciola::kernels
