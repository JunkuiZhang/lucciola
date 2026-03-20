#include "kernels/attention.h"
#include <cuda_bf16.h>
#include <math.h>

namespace lucciola::kernels {

__global__ void append_kv_cache_kernel(
    __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache,
    const __nv_bfloat16 *__restrict__ k_new,
    const __nv_bfloat16 *__restrict__ v_new,
    const int kv_seq_len,
    const int seq_len,
    const int num_kv_heads,
    const int head_dim) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; // seq_len index
    int h = blockIdx.y * blockDim.y + threadIdx.y; // head index
    int d = blockIdx.z * blockDim.z + threadIdx.z; // dim index

    if (i < seq_len && h < num_kv_heads && d < head_dim) {
        int dst_idx =
            (kv_seq_len + i) * num_kv_heads * head_dim + h * head_dim + d;
        int src_idx = i * num_kv_heads * head_dim + h * head_dim + d;
        k_cache[dst_idx] = k_new[src_idx];
        v_cache[dst_idx] = v_new[src_idx];
    }
}

// ==========================================
// Kernel 1: Q * K_cache^T
// Q: [seq_len, num_heads, head_dim]
// K_cache: [total_seq_len, num_kv_heads, head_dim]
// Score: [num_heads, seq_len, total_seq_len]
// ==========================================
__global__ void qk_matmul_kernel(
    float *__restrict__ score,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    const int seq_len,
    const int total_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim) {

    int h = blockIdx.z;                            // Per head
    int i = blockIdx.y * blockDim.y + threadIdx.y; // seq_len (Q query token)
    int j =
        blockIdx.x * blockDim.x + threadIdx.x; // total_seq_len (K key token)

    if (i < seq_len && j < total_seq_len) {
        float sum = 0.0f;
        int kv_h = h / (num_heads / num_kv_heads); // GQA mapping

        const __nv_bfloat16 *q_row =
            q + (i * num_heads * head_dim + h * head_dim);
        const __nv_bfloat16 *k_row =
            k_cache + (j * num_kv_heads * head_dim + kv_h * head_dim);

        for (int d = 0; d < head_dim; ++d) {
            sum += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
        }

        float scale = 1.0f / sqrtf((float)head_dim);
        score[h * seq_len * total_seq_len + i * total_seq_len + j] =
            sum * scale;
    }
}

// ==========================================
// Kernel 2: Causal Mask & Softmax
// Score: [num_heads, seq_len, total_seq_len]
// ==========================================
__global__ void mask_softmax_kernel(
    float *__restrict__ score,
    const int seq_len,
    const int kv_seq_len,
    const int total_seq_len,
    const int num_heads) {

    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len) {
        float *score_row =
            score + (h * seq_len * total_seq_len + i * total_seq_len);

        float max_val = -1e20f;
        for (int j = 0; j < total_seq_len; ++j) {
            if (j > kv_seq_len + i) {
                score_row[j] = -1e20f;
            }
            if (score_row[j] > max_val) {
                max_val = score_row[j];
            }
        }

        float sum_exp = 0.0f;
        for (int j = 0; j <= kv_seq_len + i && j < total_seq_len; ++j) {
            score_row[j] = expf(score_row[j] - max_val);
            sum_exp += score_row[j];
        }

        for (int j = 0; j <= kv_seq_len + i && j < total_seq_len; ++j) {
            score_row[j] /= sum_exp;
        }
    }
}

// ==========================================
// Kernel 3: SoftmaxOut * V_cache
// Out: [seq_len, num_heads, head_dim]
// ==========================================
__global__ void attn_v_matmul_kernel(
    __nv_bfloat16 *__restrict__ out,
    const float *__restrict__ score,
    const __nv_bfloat16 *__restrict__ v_cache,
    const int seq_len,
    const int kv_seq_len,
    const int total_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim) {

    int h = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;
        int kv_h = h / (num_heads / num_kv_heads);

        const float *score_row =
            score + (h * seq_len * total_seq_len + i * total_seq_len);

        for (int j = 0; j <= kv_seq_len + i && j < total_seq_len; ++j) {
            const __nv_bfloat16 *v_val =
                v_cache + (j * num_kv_heads * head_dim + kv_h * head_dim + d);
            sum += score_row[j] * __bfloat162float(*v_val);
        }

        out[i * num_heads * head_dim + h * head_dim + d] =
            __float2bfloat16(sum);
    }
}

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
    cudaStream_t stream) {

    int total_seq_len = kv_seq_len + seq_len;

    // 0. Append K/V to Cache
    dim3 cache_block(16, 8, 8);
    dim3 cache_grid(
        (seq_len + 15) / 16, (num_kv_heads + 7) / 8, (head_dim + 7) / 8);
    append_kv_cache_kernel<<<cache_grid, cache_block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(k_cache),
        reinterpret_cast<__nv_bfloat16 *>(v_cache),
        reinterpret_cast<const __nv_bfloat16 *>(k),
        reinterpret_cast<const __nv_bfloat16 *>(v),
        kv_seq_len,
        seq_len,
        num_kv_heads,
        head_dim);

    // Dynamic memory malloc
    size_t score_bytes =
        (size_t)num_heads * seq_len * total_seq_len * sizeof(float);
    float *d_score;
    cudaMalloc(&d_score, score_bytes);

    // 1. Q * K_cache^T
    dim3 block_qk(16, 16, 1);
    dim3 grid_qk((total_seq_len + 15) / 16, (seq_len + 15) / 16, num_heads);
    qk_matmul_kernel<<<grid_qk, block_qk, 0, stream>>>(
        d_score,
        reinterpret_cast<const __nv_bfloat16 *>(q),
        reinterpret_cast<const __nv_bfloat16 *>(k_cache),
        seq_len,
        total_seq_len,
        num_heads,
        num_kv_heads,
        head_dim);

    // 2. Causal Mask & Softmax
    dim3 block_sm(256, 1, 1);
    dim3 grid_sm((seq_len + 255) / 256, num_heads, 1);
    mask_softmax_kernel<<<grid_sm, block_sm, 0, stream>>>(
        d_score, seq_len, kv_seq_len, total_seq_len, num_heads);

    // 3. Score * V_cache
    dim3 block_v(16, 16, 1);
    dim3 grid_v((head_dim + 15) / 16, (seq_len + 15) / 16, num_heads);
    attn_v_matmul_kernel<<<grid_v, block_v, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        d_score,
        reinterpret_cast<const __nv_bfloat16 *>(v_cache),
        seq_len,
        kv_seq_len,
        total_seq_len,
        num_heads,
        num_kv_heads,
        head_dim);

    cudaFree(d_score);
}

// void prefill_attention_forward(
//     void *out,
//     const void *q,
//     const void *k,
//     const void *v,
//     const int seq_len,
//     const int num_heads,
//     const int num_kv_heads,
//     const int head_dim,
//     cudaStream_t stream) {

//     size_t score_bytes = (size_t)num_heads * seq_len * seq_len *
//     sizeof(float); float *d_score; cudaMalloc(&d_score, score_bytes);

//     dim3 block_qk(16, 16, 1);
//     dim3 grid_qk((seq_len + 15) / 16, (seq_len + 15) / 16, num_heads);
//     qk_matmul_kernel<<<grid_qk, block_qk, 0, stream>>>(
//         d_score,
//         reinterpret_cast<const __nv_bfloat16 *>(q),
//         reinterpret_cast<const __nv_bfloat16 *>(k),
//         seq_len,
//         seq_len,
//         num_heads,
//         num_kv_heads,
//         head_dim);

//     dim3 block_sm(256, 1, 1);
//     dim3 grid_sm((seq_len + 255) / 256, num_heads, 1);
//     mask_softmax_kernel<<<grid_sm, block_sm, 0, stream>>>(
//         d_score, seq_len, 0, seq_len, num_heads);

//     dim3 block_v(16, 16, 1);
//     dim3 grid_v((head_dim + 15) / 16, (seq_len + 15) / 16, num_heads);
//     attn_v_matmul_kernel<<<grid_v, block_v, 0, stream>>>(
//         reinterpret_cast<__nv_bfloat16 *>(out),
//         d_score,
//         reinterpret_cast<const __nv_bfloat16 *>(v),
//         seq_len,
//         0,
//         seq_len,
//         num_heads,
//         num_kv_heads,
//         head_dim);

//     cudaFree(d_score);
// }
} // namespace lucciola::kernels
