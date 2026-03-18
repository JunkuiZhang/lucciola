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
    seq->set_status(SequenceStatus::WAITING); // simplified swap -> wait
    swapped_.push_back(seq);
}

InputMetadata Scheduler::step() {
    InputMetadata meta;
    meta.is_prompt = false;

    // 1. Prioritize pulling from waiting to initiate a PREFILL step for 1
    // request
    if (!waiting_.empty()) {
        auto seq = waiting_.front();

        int num_logical_blocks = seq->get_num_logical_blocks();
        if (block_manager_.can_allocate(num_logical_blocks)) {
            waiting_.pop_front();
            for (int i = 0; i < num_logical_blocks; ++i) {
                seq->append_physical_block(block_manager_.allocate());
            }
            seq->set_status(SequenceStatus::RUNNING);
            running_.push_back(seq);

            meta.is_prompt = true;
            meta.seqs.push_back(seq);
            goto build_meta; // We only prefill one sequence per step currently
        }
    }

    // 2. If no prefill was performed, do a DECODE step for ALL running
    // sequences
    if (!running_.empty()) {
        auto running_copy = running_;
        for (auto seq : running_copy) {
            // Need a new block for the next token?
            int new_logical_blocks =
                seq->get_num_logical_blocks(); // After a token was appended
                                               // last step
            if (new_logical_blocks > seq->get_block_table().size()) {
                if (block_manager_.can_allocate(1)) {
                    seq->append_physical_block(block_manager_.allocate());
                } else {
                    // Preempt
                    _preempt(seq);
                    running_.erase(
                        std::find(running_.begin(), running_.end(), seq));
                    continue; // skip this seq
                }
            }
            meta.seqs.push_back(seq);
        }
    }

build_meta:
    // Build the InputMetadata flat structures for CUDA
    meta.num_seqs = meta.seqs.size();
    meta.max_blocks_per_seq = 0;

    for (auto seq : meta.seqs) {
        meta.max_blocks_per_seq = std::max(
            meta.max_blocks_per_seq, (int)seq->get_block_table().size());
        meta.context_lens.push_back(seq->get_len());

        if (meta.is_prompt) { // push all tokens for Prefill
            for (int tid : seq->get_token_ids()) {
                meta.input_tokens.push_back(tid);
            }
        } else { // push only the last token for Decode
            meta.input_tokens.push_back(seq->get_token_ids().back());
        }
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
