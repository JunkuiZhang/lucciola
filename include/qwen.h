#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "gpu_arena.h"
#include "safetensores.h"
#include "tokenizer.h"

namespace lucciola {

struct QwenConfig {
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int num_hidden_layers;
    int vocab_size;
};

class QwenBlock {
  public:
    QwenBlock(
        int layer_id,
        const QwenConfig &config,
        GpuArena &arena,
        const SafeTensors &safetensors,
        int max_seq_len);
    ~QwenBlock();

    QwenBlock(QwenBlock &&) noexcept = default;
    QwenBlock &operator=(QwenBlock &&) noexcept = default;
    QwenBlock(const QwenBlock &) = delete;
    QwenBlock &operator=(const QwenBlock &) = delete;

    // The grand forward pass stitching all custom CUDA kernels together
    void forward(
        void *hidden_states,
        int seq_len,
        int kv_seq_len,
        const int *pos_ids,
        cudaStream_t stream = 0);

  private:
    int layer_id_;
    int hidden_size_;
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int intermediate_size_;

    // Pointers to the model weights from SafeTensors
    Tensor<__nv_bfloat16> input_layernorm_weight_;
    Tensor<__nv_bfloat16> post_attention_layernorm_weight_;
    Tensor<__nv_bfloat16> q_proj_weight_;
    Tensor<__nv_bfloat16> k_proj_weight_;
    Tensor<__nv_bfloat16> v_proj_weight_;
    Tensor<__nv_bfloat16> o_proj_weight_;
    Tensor<__nv_bfloat16> q_norm_weight_;
    Tensor<__nv_bfloat16> k_norm_weight_;
    Tensor<__nv_bfloat16> gate_proj_weight_;
    Tensor<__nv_bfloat16> up_proj_weight_;
    Tensor<__nv_bfloat16> down_proj_weight_;

    // KV Cache
    void *k_cache_;
    void *v_cache_;

    // Intermediate activations from GpuArena
    void *q_buf_;
    void *k_buf_;
    void *v_buf_;
    void *attn_out_buf_;
    void *residual_buf_;
    void *mlp_gate_buf_;
    void *mlp_up_buf_;
    void *mlp_out_buf_;
};

class QwenModel {
  public:
    QwenModel(const std::string &model_path);
    ~QwenModel();

    int prefill(const std::vector<int> &input_ids);
    int decode(int last_token, int kv_seq_len);
    std::vector<int>
    generate(const std::vector<int> &input_ids, int max_new_tokens);
    const QwenTokenizer &get_tokenizer() const { return tokenizer_; }

  private:
    QwenConfig config_;
    QwenTokenizer tokenizer_;
    GpuArena arena_;
    std::vector<QwenBlock> blocks_;

    // Global Weights
    Tensor<__nv_bfloat16> embed_tokens_weight_;
    Tensor<__nv_bfloat16> norm_weight_;
    Tensor<__nv_bfloat16> lm_head_weight_;

    // Global generation buffers
    void *hidden_states_buf_ = nullptr;
    void *logits_buf_ = nullptr;
    int *out_token_buf_ = nullptr;
    int *pos_ids_buf_ = nullptr;   // GPU array for pos_ids
    int *input_ids_buf_ = nullptr; // Persistent GPU array for input_ids
};

} // namespace lucciola
