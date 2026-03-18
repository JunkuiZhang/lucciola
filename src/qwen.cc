#include "qwen.h"
#include <cuda_bf16.h>
#include <print>

#include "kernels/add.h"
#include "kernels/argmax.h"
#include "kernels/attention.h"
#include "kernels/embedding.h"
#include "kernels/linear.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "kernels/swiglu.h"
#include "safetensores.h"

#include <iostream>

#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace lucciola {

void print_tensor(const std::string &name, void *d_ptr, int size) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR [%s]: %s\n", name.c_str(), cudaGetErrorString(err));
    }
    std::vector<short> h_buf(std::min(size, 8));
    cudaMemcpy(
        h_buf.data(),
        d_ptr,
        h_buf.size() * sizeof(short),
        cudaMemcpyDeviceToHost);
    std::cout << name << ": ";
    for (auto v : h_buf) {
        printf("%04hX ", v);
    }
    std::cout << std::endl;
}

QwenModel::QwenModel(const std::string &model_path)
    : tokenizer_(model_path + "/vocab.json") {

    // 1. Parse config.json to fill QwenConfig
    std::string config_path = model_path + "/config.json";
    std::ifstream f(config_path);
    if (f.is_open()) {
        json data = json::parse(f);
        config_.hidden_size = data.value("hidden_size", 1536);
        config_.intermediate_size = data.value("intermediate_size", 8960);
        config_.num_attention_heads = data.value("num_attention_heads", 12);
        config_.num_key_value_heads =
            data.value("num_key_value_heads", config_.num_attention_heads);
        config_.head_dim = data.value(
            "head_dim", config_.hidden_size / config_.num_attention_heads);
        config_.num_hidden_layers = data.value("num_hidden_layers", 28);
        config_.vocab_size = data.value("vocab_size", 151936);
        std::println(
            "Loaded Config: [{}] Layers, [{}] Heads, [{}] Hidden",
            config_.num_hidden_layers,
            config_.num_attention_heads,
            config_.hidden_size);
    } else {
        std::println("Warning: Failed to open config.json. Using defaults.");
    }

    // 2. Load Safetensors
    std::string safetensors_path = model_path + "/model.safetensors";
    SafeTensors safetensors;
    bool load_success = safetensors.load(safetensors_path);
    if (!load_success) {
        std::println("Failed to load safetensores weights");
    }

    // 3. Bind Global Weights
    embed_tokens_weight_ =
        safetensors.get_tensor<__nv_bfloat16>("model.embed_tokens.weight");
    norm_weight_ = safetensors.get_tensor<__nv_bfloat16>("model.norm.weight");
    lm_head_weight_ = safetensors.get_tensor<__nv_bfloat16>("lm_head.weight");
    // Some models tie lm_head and embed_tokens
    if (!lm_head_weight_.data())
        lm_head_weight_ =
            safetensors.get_tensor<__nv_bfloat16>("model.embed_tokens.weight");

    std::vector<short> debug_emb(8);
    cudaMemcpy(
        debug_emb.data(),
        embed_tokens_weight_,
        8 * sizeof(short),
        cudaMemcpyDeviceToHost);
    std::cout << "Embed PTR " << embed_tokens_weight_ << " Data: ";
    for (auto v : debug_emb)
        printf("%04hX ", v);
    std::cout << std::endl;

    // 4. Initialize GPU Arena (e.g. 4GB for buffers)
    arena_.init(1024ULL * 1024 * 1024 * 4);
    int max_seq_len = 1024; // Hardcode max seq len for our buffers

    hidden_states_buf_ = arena_.alloc<float>(
        max_seq_len *
        config_.hidden_size); // BFloat16 but alloc space by size (using float
                              // to cover 4 bytes or short for 2)
    hidden_states_buf_ = arena_.alloc<short>(max_seq_len * config_.hidden_size);
    logits_buf_ = arena_.alloc<short>(max_seq_len * config_.vocab_size);
    out_token_buf_ = arena_.alloc<int>(max_seq_len);
    pos_ids_buf_ = arena_.alloc<int>(max_seq_len);
    input_ids_buf_ = arena_.alloc<int>(max_seq_len);

    // 5. Assemble Transformer Pipeline
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        blocks_.emplace_back(i, config_, arena_, safetensors, max_seq_len);
    }
}

QwenModel::~QwenModel() {}

std::vector<int>
QwenModel::generate(const std::vector<int> &input_ids, int max_new_tokens) {
    std::vector<int> generated_ids;
    int current_seq_len = input_ids.size();

    std::println(
        "\n[Engine] Starting Prefill Stage (Processing {} background "
        "tokens)...",
        current_seq_len);

    std::cout << "Input IDs: ";
    for (int id : input_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    int next_token = prefill(input_ids);
    generated_ids.push_back(next_token);

    // [Decode Stage] Autoregressive Generation
    std::println("[Engine] Context memorized. Entering Decode Loop...");

    for (int step = 0; step < max_new_tokens; ++step) {
        if (next_token == 151645 || next_token == 151643) { // EOS tokens
            break;
        }

        next_token = decode(next_token, current_seq_len);
        current_seq_len++;

        generated_ids.push_back(next_token);

        // Console streaming effect
        std::cout << tokenizer_.decode(next_token) << std::flush;
    }

    return generated_ids;
}

int QwenModel::prefill(const std::vector<int> &input_ids) {
    int current_seq_len = input_ids.size();
    cudaStream_t stream = 0;

    cudaMemcpy(
        input_ids_buf_,
        input_ids.data(),
        current_seq_len * sizeof(int),
        cudaMemcpyHostToDevice);

    std::vector<int> h_pos_ids(current_seq_len);
    for (int i = 0; i < current_seq_len; i++)
        h_pos_ids[i] = i;
    cudaMemcpy(
        pos_ids_buf_,
        h_pos_ids.data(),
        current_seq_len * sizeof(int),
        cudaMemcpyHostToDevice);

    // 1. Embedding
    kernels::embedding_forward<short>(
        hidden_states_buf_,
        input_ids_buf_,
        embed_tokens_weight_,
        current_seq_len,
        config_.hidden_size,
        config_.vocab_size,
        stream);

    // 2. Transformer Blocks
    for (auto &block : blocks_) {
        block.forward(
            hidden_states_buf_, current_seq_len, 0, pos_ids_buf_, stream);
    }

    // 3. Final LM Head and Argmax
    kernels::rmsnorm_forward(
        hidden_states_buf_,
        hidden_states_buf_,
        norm_weight_,
        current_seq_len,
        config_.hidden_size,
        1e-6f,
        stream);

    // For generation, we only need the LAST token's logit
    short *last_hidden_state = static_cast<short *>(hidden_states_buf_) +
                               (current_seq_len - 1) * config_.hidden_size;

    kernels::linear_forward(
        logits_buf_,
        last_hidden_state,
        lm_head_weight_,
        1,
        config_.vocab_size,
        config_.hidden_size,
        stream);

    kernels::argmax_forward(
        out_token_buf_, logits_buf_, 1, config_.vocab_size, stream);

    int next_token = 0;
    cudaMemcpy(
        &next_token, out_token_buf_, sizeof(int), cudaMemcpyDeviceToHost);

    return next_token;
}

int QwenModel::decode(int last_token, int kv_seq_len) {
    cudaStream_t stream = 0;
    int seq_len = 1;

    cudaMemcpy(
        input_ids_buf_, &last_token, sizeof(int), cudaMemcpyHostToDevice);
    int pos_id = kv_seq_len;
    cudaMemcpy(pos_ids_buf_, &pos_id, sizeof(int), cudaMemcpyHostToDevice);

    kernels::embedding_forward<short>(
        hidden_states_buf_,
        input_ids_buf_,
        embed_tokens_weight_,
        seq_len,
        config_.hidden_size,
        config_.vocab_size,
        stream);

    for (auto &block : blocks_) {
        block.forward(
            hidden_states_buf_, seq_len, kv_seq_len, pos_ids_buf_, stream);
    }

    kernels::rmsnorm_forward(
        hidden_states_buf_,
        hidden_states_buf_,
        norm_weight_,
        seq_len,
        config_.hidden_size,
        1e-6f,
        stream);

    kernels::linear_forward(
        logits_buf_,
        hidden_states_buf_,
        lm_head_weight_,
        1,
        config_.vocab_size,
        config_.hidden_size,
        stream);

    kernels::argmax_forward(
        out_token_buf_, logits_buf_, 1, config_.vocab_size, stream);

    int next_token = 0;
    cudaMemcpy(
        &next_token, out_token_buf_, sizeof(int), cudaMemcpyDeviceToHost);

    return next_token;
}

QwenBlock::QwenBlock(
    int layer_id,
    const QwenConfig &config,
    GpuArena &arena,
    const SafeTensors &safetensors,
    int max_seq_len)
    : layer_id_(layer_id), hidden_size_(config.hidden_size),
      num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads), head_dim_(config.head_dim),
      intermediate_size_(config.intermediate_size) {

    std::string base = "model.layers." + std::to_string(layer_id);

    input_layernorm_weight_ =
        safetensors.get_tensor<__nv_bfloat16>(base + ".input_layernorm.weight");
    post_attention_layernorm_weight_ = safetensors.get_tensor<__nv_bfloat16>(
        base + ".post_attention_layernorm.weight");
    q_proj_weight_ = safetensors.get_tensor<__nv_bfloat16>(
        base + ".self_attn.q_proj.weight");
    k_proj_weight_ = safetensors.get_tensor<__nv_bfloat16>(
        base + ".self_attn.k_proj.weight");
    v_proj_weight_ = safetensors.get_tensor<__nv_bfloat16>(
        base + ".self_attn.v_proj.weight");
    o_proj_weight_ = safetensors.get_tensor<__nv_bfloat16>(
        base + ".self_attn.o_proj.weight");
    q_norm_weight_ = safetensors.get_tensor<__nv_bfloat16>(
        base + ".self_attn.q_norm.weight");
    k_norm_weight_ = safetensors.get_tensor<__nv_bfloat16>(
        base + ".self_attn.k_norm.weight");
    gate_proj_weight_ =
        safetensors.get_tensor<__nv_bfloat16>(base + ".mlp.gate_proj.weight");
    up_proj_weight_ =
        safetensors.get_tensor<__nv_bfloat16>(base + ".mlp.up_proj.weight");
    down_proj_weight_ =
        safetensors.get_tensor<__nv_bfloat16>(base + ".mlp.down_proj.weight");

    // Persistent KV Cache allocating [max_seq_len, num_kv_heads, head_dim]
    k_cache_ = arena.alloc<short>(max_seq_len * num_kv_heads_ * head_dim_);
    v_cache_ = arena.alloc<short>(max_seq_len * num_kv_heads_ * head_dim_);

    // Q buf needs full [seq_len, num_heads * head_dim]
    q_buf_ = arena.alloc<short>(max_seq_len * num_heads_ * head_dim_);
    // K and V temporal buffers [seq_len, num_kv_heads * head_dim]
    k_buf_ = arena.alloc<short>(max_seq_len * num_kv_heads_ * head_dim_);
    v_buf_ = arena.alloc<short>(max_seq_len * num_kv_heads_ * head_dim_);
    attn_out_buf_ = arena.alloc<short>(max_seq_len * hidden_size_);
    residual_buf_ = arena.alloc<short>(max_seq_len * hidden_size_);
    mlp_gate_buf_ = arena.alloc<short>(max_seq_len * config.intermediate_size);
    mlp_up_buf_ = arena.alloc<short>(max_seq_len * config.intermediate_size);
    mlp_out_buf_ = arena.alloc<short>(max_seq_len * config.intermediate_size);
}

QwenBlock::~QwenBlock() {}

void QwenBlock::forward(
    void *hidden_states,
    int seq_len,
    int kv_seq_len,
    const int *pos_ids,
    cudaStream_t stream) {

    int num_tokens = seq_len;

    // Save Residual
    cudaMemcpyAsync(
        residual_buf_,
        hidden_states,
        num_tokens * hidden_size_ * sizeof(short),
        cudaMemcpyDeviceToDevice,
        stream);

    // 1. Attention Pre-Norm
    kernels::rmsnorm_forward(
        hidden_states,
        hidden_states,
        input_layernorm_weight_,
        num_tokens,
        hidden_size_,
        1e-6f,
        stream);

    // 2. QKV Projections (GQA aware)
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
        num_kv_heads_ * head_dim_,
        hidden_size_,
        stream);
    kernels::linear_forward(
        v_buf_,
        hidden_states,
        v_proj_weight_,
        num_tokens,
        num_kv_heads_ * head_dim_,
        hidden_size_,
        stream);

    // 3. QK-Norm (Qwen3: per-head RMSNorm on Q and K before RoPE)
    // Q shape: [num_tokens, num_heads, head_dim] — treat as
    // [num_tokens*num_heads, head_dim]
    kernels::rmsnorm_forward(
        q_buf_,
        q_buf_,
        q_norm_weight_,
        num_tokens * num_heads_,
        head_dim_,
        1e-6f,
        stream);
    // K shape: [num_tokens, num_kv_heads, head_dim]
    kernels::rmsnorm_forward(
        k_buf_,
        k_buf_,
        k_norm_weight_,
        num_tokens * num_kv_heads_,
        head_dim_,
        1e-6f,
        stream);

    // 4. Rotary Position Embedding (RoPE)
    kernels::rope_forward(
        q_buf_, pos_ids, num_tokens, num_heads_, head_dim_, 1000000.0f, stream);
    kernels::rope_forward(
        k_buf_,
        pos_ids,
        num_tokens,
        num_kv_heads_,
        head_dim_,
        1000000.0f,
        stream);

    // 4. Naive Multi-Head Attention with KV Cache
    kernels::kv_cache_attention_forward(
        attn_out_buf_,
        q_buf_,
        k_buf_,
        v_buf_,
        k_cache_,
        v_cache_,
        seq_len,
        kv_seq_len,
        num_heads_,
        num_kv_heads_,
        head_dim_,
        stream);

    // 5. Attention Output Projection
    kernels::linear_forward(
        hidden_states,
        attn_out_buf_,
        o_proj_weight_,
        num_tokens,
        hidden_size_,
        num_heads_ * head_dim_,
        stream);

    // 5.5 Residual Add
    kernels::add_forward(
        hidden_states, residual_buf_, num_tokens * hidden_size_, stream);

    // Save Residual for MLP
    cudaMemcpyAsync(
        residual_buf_,
        hidden_states,
        num_tokens * hidden_size_ * sizeof(short),
        cudaMemcpyDeviceToDevice,
        stream);

    // 6. MLP Pre-Norm
    kernels::rmsnorm_forward(
        hidden_states,
        hidden_states,
        post_attention_layernorm_weight_,
        num_tokens,
        hidden_size_,
        1e-6f,
        stream);

    // 7. MLP SwiGLU
    kernels::linear_forward(
        mlp_gate_buf_,
        hidden_states,
        gate_proj_weight_,
        num_tokens,
        intermediate_size_,
        hidden_size_,
        stream);
    kernels::linear_forward(
        mlp_up_buf_,
        hidden_states,
        up_proj_weight_,
        num_tokens,
        intermediate_size_,
        hidden_size_,
        stream);

    kernels::swiglu_forward(
        mlp_out_buf_,
        mlp_gate_buf_,
        mlp_up_buf_,
        num_tokens,
        intermediate_size_,
        stream);

    // 8. MLP Output Projection
    kernels::linear_forward(
        hidden_states,
        mlp_out_buf_,
        down_proj_weight_,
        num_tokens,
        hidden_size_,
        intermediate_size_,
        stream);

    // 8.5 MLP Residual Add
    kernels::add_forward(
        hidden_states, residual_buf_, num_tokens * hidden_size_, stream);
}

} // namespace lucciola