#include "sequence.h"

namespace lucciola {

Sequence::Sequence(
    int seq_id, const std::vector<int> &prompt_token_ids, int block_size)
    : seq_id_(seq_id), status_(SequenceStatus::WAITING),
      prompt_len_(prompt_token_ids.size()), prefill_offset_(0),
      block_size_(block_size), token_ids_(prompt_token_ids) {}

void Sequence::append_token_id(int token_id) { token_ids_.push_back(token_id); }

int Sequence::get_num_logical_blocks() const {
    return (token_ids_.size() + block_size_ - 1) / block_size_;
}

void Sequence::append_physical_block(int physical_block_id) {
    physical_blocks_.push_back(physical_block_id);
}

} // namespace lucciola
