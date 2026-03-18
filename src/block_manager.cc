#include "block_manager.h"
#include <iostream>

namespace lucciola {

BlockManager::BlockManager(const CacheConfig &config) : config_(config) {
    if (config.num_gpu_blocks > 0) {
        init(config.num_gpu_blocks);
    }
}

void BlockManager::init(int num_gpu_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    free_gpu_blocks_.clear();
    config_.num_gpu_blocks = num_gpu_blocks;
    for (int i = 0; i < config_.num_gpu_blocks; ++i) {
        free_gpu_blocks_.push_back(i);
    }
}

bool BlockManager::can_allocate(int num_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    return free_gpu_blocks_.size() >= static_cast<size_t>(num_blocks);
}

int BlockManager::allocate() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_gpu_blocks_.empty()) {
        std::cout << "GPU ALLOC FAILED!" << std::endl;
        return -1; // Allocation failed
    }
    int block_id = free_gpu_blocks_.front();
    free_gpu_blocks_.pop_front();
    return block_id;
}

void BlockManager::free(int physical_block_id) {
    std::lock_guard<std::mutex> lock(mu_);
    free_gpu_blocks_.push_back(physical_block_id);
}

int BlockManager::get_num_free_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return free_gpu_blocks_.size();
}

void BlockManager::init_cpu(int num_cpu_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    free_cpu_blocks_.clear();
    config_.num_cpu_blocks = num_cpu_blocks;
    for (int i = 0; i < config_.num_cpu_blocks; ++i) {
        free_cpu_blocks_.push_back(i);
    }
}

bool BlockManager::can_allocate_cpu(int num_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    return free_cpu_blocks_.size() >= static_cast<size_t>(num_blocks);
}

int BlockManager::allocate_cpu() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_cpu_blocks_.empty()) {
        std::cout << "CPU ALLOC FAILED!" << std::endl;
        return -1;
    }
    int block_id = free_cpu_blocks_.front();
    free_cpu_blocks_.pop_front();
    return block_id;
}

void BlockManager::free_cpu(int physical_block_id) {
    std::lock_guard<std::mutex> lock(mu_);
    free_cpu_blocks_.push_back(physical_block_id);
}

int BlockManager::get_num_free_cpu_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return free_cpu_blocks_.size();
}

} // namespace lucciola
