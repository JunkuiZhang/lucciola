#pragma once

#include "block_manager.h"
#include "sequence.h"
#include <deque>

namespace lucciola {

class Scheduler {
  public:
    Scheduler(const CacheConfig &env_config);

    void init(const CacheConfig &env_config);

    void add_request(SequencePtr seq);

    // Creates the input metadata for the next forward pass
    InputMetadata step();

    bool has_unfinished_requests() const;

    void free_sequence(SequencePtr seq);

  private:
    BlockManager block_manager_;
    int block_size_;

    std::deque<SequencePtr> waiting_;
    std::deque<SequencePtr> running_;
    std::deque<SequencePtr> swapped_;

    void _preempt(SequencePtr seq);
};

} // namespace lucciola
