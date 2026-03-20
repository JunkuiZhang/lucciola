#include "scheduler.h"
#include <algorithm>

namespace lucciola {

Scheduler::Scheduler(const CacheConfig &env_config)
    : block_manager_(env_config), block_size_(env_config.block_size) {}

void Scheduler::init(const CacheConfig &env_config) {
    block_manager_.init(env_config.num_gpu_blocks);
    block_size_ = env_config.block_size;
}

void Scheduler::add_request(SequencePtr seq) { waiting_.push_back(seq); }

bool Scheduler::has_unfinished_requests() const {
    return !waiting_.empty() || !running_.empty() || !swapped_.empty();
}

void Scheduler::free_sequence(SequencePtr seq) {
    for (int block_id : seq->get_block_table()) {
        block_manager_.free(block_id);
    }
    seq->clear_blocks();
    seq->set_status(SequenceStatus::FINISHED);

    auto it = std::find(running_.begin(), running_.end(), seq);
    if (it != running_.end()) {
        running_.erase(it);
    }
}

void Scheduler::_preempt(SequencePtr seq) {
    for (int block_id : seq->get_block_table()) {
        block_manager_.free(block_id);
    }
    seq->clear_blocks();
    seq->set_status(SequenceStatus::WAITING);
    swapped_.push_back(seq);
}

InputMetadata Scheduler::step() {
    InputMetadata meta;
    meta.num_decode_seqs = 0;
    meta.num_prefill_seqs = 0;
    meta.num_prefill_tokens = 0;

    const int MAX_TOKENS_PER_STEP = 512 + 64;
    int total_tokens_this_step = 0;

    // 1. Schedule running requests (Decode) first
    if (!running_.empty()) {
        auto running_copy = running_;
        for (auto seq : running_copy) {
            int new_logical_blocks = seq->get_num_logical_blocks();
            if (new_logical_blocks > seq->get_block_table().size()) {
                if (block_manager_.can_allocate(1)) {
                    seq->append_physical_block(block_manager_.allocate());
                } else {
                    _preempt(seq);
                    running_.erase(
                        std::find(running_.begin(), running_.end(), seq));
                    continue;
                }
            }
            meta.seqs.push_back(seq);
            meta.context_lens.push_back(seq->get_len());
            meta.input_tokens.push_back(seq->get_token_ids().back());
            meta.input_pos.push_back(
                seq->get_len() - 1); // absolute position for decode
            meta.num_decode_seqs++;
            total_tokens_this_step++;
        }
    }

    // 2. Schedule a chunked prefill request if we have enough token budget
    if (!waiting_.empty() && total_tokens_this_step < MAX_TOKENS_PER_STEP) {
        auto seq = waiting_.front();
        int token_budget = MAX_TOKENS_PER_STEP - total_tokens_this_step;

        int chunk_size = std::min(
            token_budget, seq->get_prompt_len() - seq->get_prefill_offset());

        // If chunk_size > 0, we can schedule it. We could limit scheduling to
        // e.g. chunk_size > 1 to avoid tiny chunks, but >0 is fine.
        if (chunk_size > 0) {
            int total_len = seq->get_prefill_offset() + chunk_size;

            int required_blocks = (total_len + block_size_ - 1) / block_size_;
            int blocks_to_allocate =
                required_blocks - seq->get_block_table().size();

            if (blocks_to_allocate <= 0 ||
                block_manager_.can_allocate(blocks_to_allocate)) {
                for (int i = 0; i < blocks_to_allocate; ++i) {
                    seq->append_physical_block(block_manager_.allocate());
                }

                meta.seqs.push_back(seq);
                meta.context_lens.push_back(total_len);

                auto const &tokens = seq->get_token_ids();
                for (int i = seq->get_prefill_offset(); i < total_len; ++i) {
                    meta.input_tokens.push_back(tokens[i]);
                    meta.input_pos.push_back(i); // absolute position in prompt
                }

                meta.num_prefill_seqs++;
                meta.num_prefill_tokens = chunk_size;

                seq->set_prefill_offset(total_len);

                if (seq->get_prefill_offset() >= seq->get_prompt_len()) {
                    waiting_.pop_front();
                    seq->set_status(SequenceStatus::RUNNING);
                    running_.push_back(seq);
                }
            }
        }
    }

    meta.num_seqs = meta.seqs.size();
    meta.max_blocks_per_seq = 0;

    for (auto seq : meta.seqs) {
        meta.max_blocks_per_seq = std::max(
            meta.max_blocks_per_seq, (int)seq->get_block_table().size());
    }

    meta.block_tables.resize(meta.num_seqs * meta.max_blocks_per_seq, -1);
    for (int i = 0; i < meta.num_seqs; ++i) {
        auto const &table = meta.seqs[i]->get_block_table();
        for (size_t j = 0; j < table.size(); ++j) {
            meta.block_tables[i * meta.max_blocks_per_seq + j] = table[j];
        }
    }

    return meta;
}

} // namespace lucciola
