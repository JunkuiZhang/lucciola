#include "async_engine.h"

namespace lucciola {

AsyncEngine::AsyncEngine(
    const std::string &model_path, int max_blocks, int block_size)
    : model_(model_path), cache_config_{block_size, max_blocks, 0},
      scheduler_(cache_config_), stop_flag_(false), next_req_id_(1) {

    if (max_blocks <= 0) {
        // Auto-allocate vLLM style (e.g. 90% of free VRAM)
        int calculated_blocks =
            model_.init_paged_kv_cache_auto(block_size, 0.90f);
        cache_config_.num_gpu_blocks = calculated_blocks;
        scheduler_.init(cache_config_);
    } else {
        model_.init_paged_kv_cache(max_blocks, block_size);
        scheduler_.init(cache_config_);
    }
    background_thread_ = std::thread(&AsyncEngine::engine_loop, this);
}

AsyncEngine::~AsyncEngine() { stop(); }

void AsyncEngine::stop() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stop_flag_)
            return;
        stop_flag_ = true;
    }
    cv_.notify_all();
    if (background_thread_.joinable()) {
        background_thread_.join();
    }
}

int AsyncEngine::generate_async(
    const std::string &prompt,
    int max_new_tokens,
    std::function<void(const std::string &, bool)> callback) {
    std::lock_guard<std::mutex> lock(mu_);
    int req_id = next_req_id_++;

    RequestInfo info;
    info.max_new_tokens = max_new_tokens;
    info.stream_cb = std::move(callback);
    active_requests_[req_id] = std::move(info);

    pending_prompts_.push_back({req_id, prompt});
    cv_.notify_one();

    return req_id;
}

void AsyncEngine::engine_loop() {
    while (true) {
        std::vector<std::pair<int, std::string>> local_prompts;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this]() {
                return stop_flag_ || !pending_prompts_.empty() ||
                       scheduler_.has_unfinished_requests();
            });

            if (stop_flag_ && pending_prompts_.empty() &&
                !scheduler_.has_unfinished_requests()) {
                if (!local_prompts.empty())
                    printf("Received %zu prompts\n", local_prompts.size());
                break;
            }

            // Transfer pending prompts
            local_prompts = std::move(pending_prompts_);
            pending_prompts_.clear();
        }

        // 1. Process new prompts and add to scheduler
        for (auto &pair : local_prompts) {
            int req_id = pair.first;

            // Format standard chat prompt (for simplicity here, though could be
            // abstracted out)
            const int IM_START = 151644;
            const int IM_END = 151645;
            const int ENTER_ID = 1699;

            std::vector<int> input_ids;
            input_ids.push_back(IM_START);
            auto sys_ids = model_.get_tokenizer().encode(
                "system\nYou are a helpful assistant.");
            input_ids.insert(input_ids.end(), sys_ids.begin(), sys_ids.end());
            input_ids.push_back(IM_END);
            input_ids.push_back(ENTER_ID);

            input_ids.push_back(IM_START);
            auto user_ids =
                model_.get_tokenizer().encode("user\n" + pair.second);
            input_ids.insert(input_ids.end(), user_ids.begin(), user_ids.end());
            input_ids.push_back(IM_END);
            input_ids.push_back(ENTER_ID);

            input_ids.push_back(IM_START);
            auto assistant_ids = model_.get_tokenizer().encode("assistant\n");
            input_ids.insert(
                input_ids.end(), assistant_ids.begin(), assistant_ids.end());

            auto seq = std::make_shared<Sequence>(
                req_id, input_ids, cache_config_.block_size);
            scheduler_.add_request(seq);
        }

        // 2. Perform a step if there are requests
        if (scheduler_.has_unfinished_requests()) {
            InputMetadata meta = scheduler_.step();

            if (meta.num_seqs > 0) {
                // Forward pass using PagedAttention
                model_.step_with_paged_attention(meta);

                // Process output tokens
                // For each sequence in the batch, a new token was appended
                for (int i = 0; i < meta.num_seqs; ++i) {
                    auto seq = meta.seqs[i];
                    int req_id = seq->get_seq_id();

                    bool is_prefilling =
                        (i >= meta.num_decode_seqs) &&
                        (seq->get_prefill_offset() < seq->get_prompt_len());
                    if (is_prefilling) {
                        continue; // Skip processing completely if still
                                  // chunking prompt
                    }

                    int next_token = seq->get_token_ids().back();
                    bool is_finished = false;
                    std::string text_chunk = "";

                    text_chunk = model_.get_tokenizer().decode(next_token);
                    active_requests_[req_id].generated_tokens++;

                    // Check stop condition
                    if (next_token == 151645 || next_token == 151643 ||
                        active_requests_[req_id].generated_tokens >=
                            active_requests_[req_id].max_new_tokens) {
                        is_finished = true;
                    }

                    // Callback
                    if (active_requests_[req_id].stream_cb) {
                        active_requests_[req_id].stream_cb(
                            text_chunk, is_finished);
                    }

                    if (is_finished) {
                        scheduler_.free_sequence(seq);
                        std::lock_guard<std::mutex> lock(mu_);
                        active_requests_.erase(req_id);
                    }
                }
            } else {
                // Wait queue empty but maybe swapped. MVP assumes no swapped
                // yet. Or blocks exhausted, so we yield slightly?
                std::this_thread::yield();
            }
        }
    }
}

} // namespace lucciola
