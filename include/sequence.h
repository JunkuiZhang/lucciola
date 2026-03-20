#pragma once

#include <memory>
#include <vector>

namespace lucciola {

enum class SequenceStatus { WAITING, RUNNING, SWAPPED, FINISHED };

class Sequence {
  public:
    Sequence(
        int seq_id, const std::vector<int> &prompt_token_ids, int block_size);

    int get_seq_id() const { return seq_id_; }
    SequenceStatus get_status() const { return status_; }
    void set_status(SequenceStatus status) { status_ = status; }

    int get_len() const { return token_ids_.size(); }
    int get_prompt_len() const { return prompt_len_; }
    const std::vector<int> &get_token_ids() const { return token_ids_; }

    int get_prefill_offset() const { return prefill_offset_; }
    void set_prefill_offset(int offset) { prefill_offset_ = offset; }

    void append_token_id(int token_id);

    // Block logic
    int get_num_logical_blocks() const;
    void append_physical_block(int physical_block_id);
    const std::vector<int> &get_block_table() const { return physical_blocks_; }
    void clear_blocks() { physical_blocks_.clear(); }

  private:
    int seq_id_;
    SequenceStatus status_;
    int prompt_len_;
    int prefill_offset_;
    int block_size_;
    std::vector<int> token_ids_;
    std::vector<int> physical_blocks_;
};

using SequencePtr = std::shared_ptr<Sequence>;

struct InputMetadata {
    std::vector<int>
        input_tokens; // The tokens to actually compute in this step
    std::vector<int> context_lens; // Context length for each seq in the batch
    std::vector<int> block_tables; // Flattened block tables for PagedAttention
                                   // [batch_size * max_blocks]
    int max_blocks_per_seq;
    int num_seqs;
    std::vector<SequencePtr> seqs; // Reference to running sequences

    int num_decode_seqs = 0;
    int num_prefill_seqs = 0;
    int num_prefill_tokens = 0;
};

} // namespace lucciola
