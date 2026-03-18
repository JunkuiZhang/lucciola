#include "kernels/attention.h"
#include <cuda_bf16.h>
#include <math.h>

namespace lucciola::kernels {

// ==========================================
// Kernel 1: Q * K^T
// Q: [seq_len, num_heads, head_dim]
// K: [seq_len, num_heads, head_dim]
// Score: [num_heads, seq_len, seq_len]
// ==========================================
__global__ void qk_matmul_kernel(
    float *__restrict__ score,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const int seq_len,
    const int num_heads,
    const int head_dim) {

    int h = blockIdx.z;                            // Per head
    int i = blockIdx.y * blockDim.y + threadIdx.y; // seq_len (Q query token)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // seq_len (K key token)

    if (i < seq_len && j < seq_len) {
        float sum = 0.0f;
        // Pointer offset for [seq_i, h, :]
        const __nv_bfloat16 *q_row =
            q + (i * num_heads * head_dim + h * head_dim);
        // Pointer offset for [seq_j, h, :]
        const __nv_bfloat16 *k_row =
            k + (j * num_heads * head_dim + h * head_dim);

        // Naive dot product
        for (int d = 0; d < head_dim; ++d) {
            sum += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
        }

        // Scale down to prevent gradient/value explosion
        float scale = 1.0f / sqrtf((float)head_dim);

        // Write to Intermediate Float Memory [h, i, j]
        score[h * seq_len * seq_len + i * seq_len + j] = sum * scale;
    }
}

// ==========================================
// Kernel 2: Causal Mask & Softmax
// Score: [num_heads, seq_len, seq_len]
// ==========================================
__global__ void mask_softmax_kernel(
    float *__restrict__ score, const int seq_len, const int num_heads) {

    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row representing Q token

    if (i < seq_len) {
        float *score_row = score + (h * seq_len * seq_len + i * seq_len);

        // Step 2.1: Causal Mask and find numerical Maximum
        float max_val = -1e20f;
        for (int j = 0; j < seq_len; ++j) {
            // Causal rule: token i cannot see future token j (where j > i)
            if (j > i) {
                score_row[j] = -1e20f;
            }
            if (score_row[j] > max_val) {
                max_val = score_row[j];
            }
        }

        // Step 2.2: Compute Exponential and Denominator Sum
        float sum_exp = 0.0f;
        for (int j = 0; j <= i; ++j) { // Safe to ignore j > i
            score_row[j] = expf(score_row[j] - max_val);
            sum_exp += score_row[j];
        }

        // Step 2.3: Normalize to Probabilities
        for (int j = 0; j <= i; ++j) {
            score_row[j] /= sum_exp;
        }
    }
}

// ==========================================
// Kernel 3: SoftmaxOut * V
// Score: [num_heads, seq_len, seq_len]
// V: [seq_len, num_heads, head_dim]
// Out: [seq_len, num_heads, head_dim]
// ==========================================
__global__ void attn_v_matmul_kernel(
    __nv_bfloat16 *__restrict__ out,
    const float *__restrict__ score,
    const __nv_bfloat16 *__restrict__ v,
    const int seq_len,
    const int num_heads,
    const int head_dim) {

    int h = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y; // seq_len output
    int d = blockIdx.x * blockDim.x + threadIdx.x; // head dimension

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;
        const float *score_row = score + (h * seq_len * seq_len + i * seq_len);

        for (int j = 0; j <= i; ++j) { // only aggregate valid K keys up to i
            const __nv_bfloat16 *v_val =
                v + (j * num_heads * head_dim + h * head_dim + d);
            sum += score_row[j] * __bfloat162float(*v_val);
        }

        // Write resulting context vector representation
        out[i * num_heads * head_dim + h * head_dim + d] =
            __float2bfloat16(sum);
    }
}

void naive_attention_forward(
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    cudaStream_t stream) {

    // ALlocate a massive intermediate Attention Score matrix on the GPU.
    // WARNING: In production models, this scales by O(L^2) and OOMs fast.
    // Advanced systems fuse these 3 stages in SRAM (FlashAttention).
    size_t score_bytes = (size_t)num_heads * seq_len * seq_len * sizeof(float);
    float *d_score;
    cudaMalloc(&d_score, score_bytes);

    // 1. Q * K^T
    dim3 block_qk(16, 16, 1);
    dim3 grid_qk((seq_len + 15) / 16, (seq_len + 15) / 16, num_heads);
    qk_matmul_kernel<<<grid_qk, block_qk, 0, stream>>>(
        d_score,
        reinterpret_cast<const __nv_bfloat16 *>(q),
        reinterpret_cast<const __nv_bfloat16 *>(k),
        seq_len,
        num_heads,
        head_dim);

    // 2. Causal Masking & Softmax
    dim3 block_sm(256, 1, 1);
    dim3 grid_sm((seq_len + 255) / 256, num_heads, 1);
    mask_softmax_kernel<<<grid_sm, block_sm, 0, stream>>>(
        d_score, seq_len, num_heads);

    // 3. Score * V = Output Out
    dim3 block_v(16, 16, 1);
    dim3 grid_v((head_dim + 15) / 16, (seq_len + 15) / 16, num_heads);
    attn_v_matmul_kernel<<<grid_v, block_v, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        d_score,
        reinterpret_cast<const __nv_bfloat16 *>(v),
        seq_len,
        num_heads,
        head_dim);

    // Educational flush
    cudaFree(d_score);
}

} // namespace lucciola::kernels
