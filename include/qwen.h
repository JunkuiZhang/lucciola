#pragma once

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

    // The grand forward pass stitching all custom CUDA kernels together
    void forward(
        void *hidden_states,
        int seq_len,
        const int *pos_ids,
        cudaStream_t stream = 0);

  private:
    int layer_id_;
    int hidden_size_;
    int num_heads_;
    int head_dim_;

    // Pointers to the model weights from SafeTensors
    const void *input_layernorm_weight_;
    const void *post_attention_layernorm_weight_;
    const void *q_proj_weight_;
    const void *k_proj_weight_;
    const void *v_proj_weight_;
    const void *o_proj_weight_;
    const void *gate_proj_weight_;
    const void *up_proj_weight_;
    const void *down_proj_weight_;

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

    std::vector<int>
    generate(const std::vector<int> &input_ids, int max_new_tokens);
    const QwenTokenizer &get_tokenizer() const { return tokenizer_; }

  private:
    QwenConfig config_;
    QwenTokenizer tokenizer_;
    SafeTensors safetensors_;
    GpuArena arena_;
    std::vector<QwenBlock> blocks_;

    // Global Weights
    const void *embed_tokens_weight_ = nullptr;
    const void *norm_weight_ = nullptr;
    const void *lm_head_weight_ = nullptr;

    // Global generation buffers
    void *hidden_states_buf_ = nullptr;
    void *logits_buf_ = nullptr;
    int *out_token_buf_ = nullptr;
    int *pos_ids_buf_ = nullptr; // GPU array for pos_ids
};

} // namespace lucciola
