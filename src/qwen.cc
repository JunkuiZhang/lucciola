#include "qwen.h"
#include "safetensores.h"
#include <print>

#include "kernels/attention.h"
#include "kernels/linear.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "kernels/swiglu.h"

namespace lucciola {

QwenModel::QwenModel(const std::string &model_path) {
    std::string safetensors_path = model_path + "/model.safetensors";
    auto safetensores = SafeTensors();
    bool load_success = safetensores.load(safetensors_path);
    if (!load_success) {
        std::println("Failed to load safetensores");
    }
}

QwenModel::~QwenModel() {}

QwenBlock::QwenBlock(int layer_id, int hidden_size, int num_heads, int head_dim)
    : layer_id_(layer_id), hidden_size_(hidden_size), num_heads_(num_heads),
      head_dim_(head_dim) {
    // Note: In reality, allocate the _buf variables with cudaMalloc
    // and map the _weight variables to specific loaded .safetensors pointers.
}

QwenBlock::~QwenBlock() {}

void QwenBlock::forward(
    void *hidden_states, int seq_len, const int *pos_ids, cudaStream_t stream) {

    int num_tokens =
        seq_len; // Assuming a flat single batch for this illustration

    // 1. Attention Pre-Norm
    // Ideally we save the residual here (e.g., cudaMemcpyAsync to
    // residual_buf_)
    kernels::rmsnorm_forward(
        hidden_states, // In-place normalization
        hidden_states,
        input_layernorm_weight_,
        num_tokens,
        hidden_size_,
        1e-6f,
        stream);

    // 2. QKV Projections
    kernels::linear_forward(
        q_buf_,
        hidden_states,
        q_proj_weight_,
        num_tokens,
        num_heads_ * head_dim_,
        hidden_size_,
        stream);
    kernels::linear_forward(
        k_buf_,
        hidden_states,
        k_proj_weight_,
        num_tokens,
        num_heads_ * head_dim_,
        hidden_size_,
        stream);
    kernels::linear_forward(
        v_buf_,
        hidden_states,
        v_proj_weight_,
        num_tokens,
        num_heads_ * head_dim_,
        hidden_size_,
        stream);

    // 3. Rotary Position Embedding (RoPE)
    // Applies the positional sine/cosine rotation in-place
    kernels::rope_forward(
        q_buf_, pos_ids, num_tokens, num_heads_, head_dim_, 1000000.0f, stream);
    kernels::rope_forward(
        k_buf_, pos_ids, num_tokens, num_heads_, head_dim_, 1000000.0f, stream);

    // 4. Naive Multi-Head Attention -> Out
    kernels::naive_attention_forward(
        attn_out_buf_,
        q_buf_,
        k_buf_,
        v_buf_,
        seq_len,
        num_heads_,
        head_dim_,
        stream);

    // 5. Attention Output Projection & Residual Add
    kernels::linear_forward(
        hidden_states,
        attn_out_buf_,
        o_proj_weight_,
        num_tokens,
        hidden_size_,
        num_heads_ * head_dim_,
        stream);
    // (Here we would element-wise add the residual_buf_ back into
    // hidden_states)

    // 6. MLP Pre-Norm
    // (Save residual_buf_ again)
    kernels::rmsnorm_forward(
        hidden_states,
        hidden_states,
        post_attention_layernorm_weight_,
        num_tokens,
        hidden_size_,
        1e-6f,
        stream);

    // 7. MLP SwiGLU
    int intermediate_size = 13696; // Standard expansion ratio for Qwen
    kernels::linear_forward(
        mlp_gate_buf_,
        hidden_states,
        gate_proj_weight_,
        num_tokens,
        intermediate_size,
        hidden_size_,
        stream);
    kernels::linear_forward(
        mlp_up_buf_,
        hidden_states,
        up_proj_weight_,
        num_tokens,
        intermediate_size,
        hidden_size_,
        stream);

    // out = SiLU(gate) * up
    kernels::swiglu_forward(
        mlp_out_buf_,
        mlp_gate_buf_,
        mlp_up_buf_,
        num_tokens,
        intermediate_size,
        stream);

    // 8. MLP Output Projection & Residual Add
    kernels::linear_forward(
        hidden_states,
        mlp_out_buf_,
        down_proj_weight_,
        num_tokens,
        hidden_size_,
        intermediate_size,
        stream);
    // (Final element-wise add of residual_buf_ back to hidden_states)

    // A single layer's logical journey is perfectly complete!
}

} // namespace lucciola