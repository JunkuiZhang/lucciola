#pragma once

#include <deque>
#include <mutex>

namespace lucciola {

struct CacheConfig {
    int block_size;
    int num_gpu_blocks;
    int num_cpu_blocks;
};

class BlockManager {
  public:
    BlockManager(const CacheConfig &config);
    ~BlockManager() = default;

    void init(int num_gpu_blocks);
    void init_cpu(int num_cpu_blocks);

    bool can_allocate(int num_blocks);
    int allocate();
    void free(int physical_block_id);
    int get_num_free_blocks() const;

    bool can_allocate_cpu(int num_blocks);
    int allocate_cpu();
    void free_cpu(int physical_block_id);
    int get_num_free_cpu_blocks() const;

  private:
    CacheConfig config_;
    std::deque<int> free_gpu_blocks_;
    std::deque<int> free_cpu_blocks_;
    mutable std::mutex mu_;
};

} // namespace lucciola
