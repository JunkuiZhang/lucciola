#pragma once

#include <string>
#include <cuda_runtime.h>
namespace lucciola {

class QwenModel {
  public:
    QwenModel(const std::string &model_path);
    ~QwenModel();

  private:
};

class QwenBlock {
  public:
    QwenBlock(int layer_id, int hidden_size, int num_heads, int head_dim);
    ~QwenBlock();

    // The grand forward pass stitching all custom CUDA kernels together
    void forward(void *hidden_states, int seq_len, const int *pos_ids, cudaStream_t stream = 0);

  private:
    int layer_id_;
    int hidden_size_;
    int num_heads_;
    int head_dim_;

    // Pointers to the model weights (would point to SafeTensors memory)
    void *input_layernorm_weight_;
    void *post_attention_layernorm_weight_;
    void *q_proj_weight_;
    void *k_proj_weight_;
    void *v_proj_weight_;
    void *o_proj_weight_;
    void *gate_proj_weight_;
    void *up_proj_weight_;
    void *down_proj_weight_;

    // Intermediate activations logic (would be grabbed from a GPU memory pool)
    void *q_buf_;
    void *k_buf_;
    void *v_buf_;
    void *attn_out_buf_;
    void *residual_buf_;
    void *mlp_gate_buf_;
    void *mlp_up_buf_;
    void *mlp_out_buf_;
};

} // namespace lucciola
