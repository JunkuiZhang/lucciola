#include <cfloat>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffff

namespace lucciola::kernels {

// 帮手 1：Warp 级别的求和归约 (我已经帮你写好了，直接抄)
__inline__ __device__ float warpReduceSum(float val) {
    // 每次折半交换并相加：16 -> 8 -> 4 -> 2 -> 1
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
    // 运行结束后，Warp 里第 0 号线程（Lane 0）的 val 就是 32 个线程的总和
}

// 帮手 2：Warp 级别的最大值归约
__inline__ __device__ float warpReduceMax(float val) {
    // ==========================================
    // TODO: 请模仿上面的 warpReduceSum，用 __shfl_down_sync 和 fmaxf
    // 写一个 5 次循环的树状规约，找出最大值。
    // ==========================================
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }

    return val;
}

// 假设我们每个物理页 (Block) 存 16 个 Token
#define PAGE_SIZE 16

// q_len: 当前切片的长度 (如果是 Decode 就是 1，如果是 Prefill 可能是 256)
// context_len: 这个请求截止到目前为止的历史总长度 (包含当前切片)
// head_dim: 注意力头的维度 (比如 128)
__global__ void chunked_paged_attention_kernel(
    float *out,                  // [num_seqs, num_heads, head_dim]
    const float *__restrict__ q, // [num_seqs, q_len, num_heads, head_dim]
    const float
        *__restrict__ k_cache, // [num_blocks, num_heads, PAGE_SIZE, head_dim]
    const float
        *__restrict__ v_cache, // [num_blocks, num_heads, PAGE_SIZE, head_dim]
    const int *__restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int *__restrict__ context_lens, // [num_seqs]
    const int *__restrict__ q_lens, // [num_seqs]  <- 异构的核心！每个序列的 Q
                                    // 长度不同
    int max_blocks_per_seq,
    int head_dim,
    float scale) // 1.0 / sqrt(head_dim)
{
    // 1. 灵魂三问：我在哪？
    // 假设 1 个 Block 负责处理 1 个序列 (Seq) 的 1 个头 (Head) 的 1 个 Query
    // Token
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.x;
    int q_token_idx =
        threadIdx.y; // 当前处理的是这个切片里的第几个 Q Token (0 到 q_len-1)
    int lane_id =
        threadIdx.x; // 用于在 head_dim (128) 上做并行规约，通常是一个 Warp (32)

    int q_len = q_lens[seq_idx];
    if (q_token_idx >= q_len)
        return;

    int context_len = context_lens[seq_idx];

    // 【极其重要】：计算当前这个 Q Token 在整个完整句子中的绝对位置！
    // 比如 context_len=1000, q_len=256。说明当前切片是第 744~999 个 Token。
    // 如果 q_token_idx = 0，那它的绝对位置就是 744。
    int absolute_q_pos = context_len - q_len + q_token_idx;

    // 定位当前 Q Token 的内存起始指针
    // 简单起见，假设 Q 已经被展平，按照 [seq_idx, q_token_idx, head_idx,
    // head_dim] 排列 （实际工程中偏移量计算会更复杂，我们先用伪代码抽象概念）
    int num_heads = blockDim.x;
    const float *my_q =
        q + (seq_idx * q_len * num_heads * head_dim +
             q_token_idx * num_heads * head_dim + head_idx * head_dim);

    // 获取当前序列的页表指针
    const int *my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // 申请一块寄存器或共享内存来存分数 (这里为了简单，我们暂存到共享内存)
    // 假设最大 context_len 是 4096
    __shared__ float logits[4096];

    // =========================================================
    // TODO 1: 穿越页表，计算 Q * K^T
    // =========================================================
    // 我们需要用当前的 my_q，去和历史中 [0, context_len-1] 的所有 K 算点积。

    // 外层循环：遍历历史上所有的 Token (从 0 到 context_len - 1)
    for (int kv_idx = 0; kv_idx < context_len; ++kv_idx) {

        // =====================================================
        // 关键挑战 A：寻址魔法 (Paged Memory Access)
        // 这个 kv_idx 到底存在哪个物理页 (Physical Block) 的哪个偏移 (Offset)？
        // =====================================================
        int logical_page_num = kv_idx / PAGE_SIZE;
        int offset_in_page = kv_idx % PAGE_SIZE;
        int physical_block_id = my_block_table[logical_page_num];

        // 找到那个特定的 K 的指针
        const float *target_k =
            k_cache + physical_block_id * (num_heads * PAGE_SIZE * head_dim) +
            head_idx * (PAGE_SIZE * head_dim) + offset_in_page * head_dim;

        // =====================================================
        // 关键挑战 B：因果掩码 (Causal Mask)
        // =====================================================
        // 如果当前遍历到的 kv_idx 大于了当前 Q Token 的绝对位置
        // (absolute_q_pos) 说明这是“未来的词”，绝对不能看！ 请直接将分数置为
        // -FLT_MAX，然后 continue 跳过计算。

        if (kv_idx > absolute_q_pos) {
            if (lane_id == 0) {
                logits[kv_idx] = -FLT_MAX; // 这个位置打上负无穷的掩码
            }
            continue; // 跳过后续的点积计算
        }

        // =====================================================
        // 关键挑战 C：计算点积分数 (Dot Product)
        // =====================================================
        // 用当前线程 (lane_id) 配合 Warp 级规约，计算 my_q 和 target_k 的点积。
        // 注意：这里需要你上节课写的 warpReduceSum 技巧！

        float score = 0.0f;
        for (int i = lane_id; i < head_dim; i += 32) {
            score += my_q[i] * target_k[i];
        }
        score = warpReduceSum(score);

        if (lane_id == 0) {
            logits[kv_idx] = score * scale;
        }
    }

    __syncthreads();

    // 定位到当前 Q Token 专属的分数数组行
    // (为了简化，假设共享内存是一个二维数组 logits_smem[q_len][4096])
    // 假设一个 Block 最多处理 32 个 Q Token
    __shared__ float logits_smem[32][4096];

    // 然后在代码里，每个 Warp 只拿属于自己的那一行：
    float *my_logits = logits_smem[q_token_idx];
    // =========================================================
    // TODO 2: 对 my_logits 数组进行 Warp 级 Softmax
    // =========================================================
    float thread_max = -FLT_MAX;

    // 步骤 A：跨步循环 (Stride Loop)，找出局部的 thread_max
    // 提示：遍历 my_logits，范围是 0 到 context_len - 1，步长为 32
    for (int index = lane_id; index < context_len; index += 32) {
        thread_max = fmaxf(thread_max, my_logits[index]);
    }

    // 步骤 B：Warp 规约找出 global_max，并由 0 号线程广播给所有人
    float global_max = warpReduceMax(thread_max);
    global_max = __shfl_sync(FULL_MASK, global_max, 0);

    // 步骤 C：跨步循环，计算局部的指数和 thread_sum
    // 注意：用 __expf(my_logits[i] - global_max)
    float thread_sum = 0.0f;
    for (int index = lane_id; index < context_len; index += 32) {
        thread_sum += __expf(my_logits[index] - global_max);
    }

    // 步骤 D：Warp 规约找出 global_sum，并由 0 号线程广播给所有人
    float global_sum = warpReduceSum(thread_sum);
    global_sum = __shfl_sync(FULL_MASK, global_sum, 0);

    // 步骤 E：跨步循环，计算最终概率并写回 my_logits
    for (int index = lane_id; index < context_len; index += 32) {
        my_logits[index] = __expf(my_logits[index] - global_max) / global_sum;
    }

    // 等待所有 Warp 把自己的 Softmax 算完
    __syncthreads();

    // =========================================================
    // TODO 3: 穿越页表，计算 P * V，并写回全局显存 (Global Memory)
    // =========================================================

    // 1. 为当前线程准备寄存器累加器
    // 假设 head_dim 最大为 128。一个 Warp 32 个人，每人最多负责 128/32 = 4
    // 个元素。 我们用一个局部的寄存器数组来存这 4 个维度的累加结果。
    float acc_v[4] = {0.0f};

    // 2. 再次穿越历史长河 (遍历所有的 K/V Token)
    for (int kv_idx = 0; kv_idx < context_len; ++kv_idx) {

        // 拿到刚刚算好的概率权重
        float p = my_logits[kv_idx];

        // 【极客小优化】：如果概率由于 Causal Mask 变成了
        // 0，直接跳过，省下巨量访存！
        if (p == 0.0f)
            continue;

        // =====================================================
        // 挑战 A：寻址魔法 (重演)
        // =====================================================
        // 请模仿阶段 1，计算 logical_page_num, offset_in_page,
        // physical_block_id 然后找到 target_v 的指针！(别忘了 head_idx
        // 维度的偏移)

        int logical_page_num = kv_idx / PAGE_SIZE;
        int offset_in_page = kv_idx % PAGE_SIZE;
        int physical_block_id = my_block_table[logical_page_num];
        const float *target_v =
            v_cache + physical_block_id * (num_heads * PAGE_SIZE * head_dim) +
            head_idx * (PAGE_SIZE * head_dim) + offset_in_page * head_dim;

        // =====================================================
        // 挑战 B：向量标量乘加 (FMA)
        // =====================================================
        // 用 p 乘以 target_v 的对应维度，累加到 acc_v 中。
        // 注意：用一个变量 arr_idx 追踪当前写到了 acc_v 的第几个槽位
        int arr_idx = 0;
        for (int i = lane_id; i < head_dim; i += 32) {
            acc_v[arr_idx] += p * target_v[i];
            arr_idx++;
        }
    }

    // =====================================================
    // 挑战 C：最终胜利的写回！
    // =====================================================
    // 定位输出矩阵 out 的指针 (假设形状也是 [num_seqs, q_len, num_heads,
    // head_dim]) 请模仿 my_q 的偏移量计算公式，算出 my_out 的指针。

    float *my_out =
        out + (seq_idx * q_len * num_heads * head_dim +
               q_token_idx * num_heads * head_dim + head_idx * head_dim);

    // 最后一次跨步循环，把寄存器里的结果写回显存
    int arr_idx = 0;
    for (int i = lane_id; i < head_dim; i += 32) {
        my_out[i] = acc_v[arr_idx++];
    }
}

} // namespace lucciola::kernels