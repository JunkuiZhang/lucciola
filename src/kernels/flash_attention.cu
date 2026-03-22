#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>

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

#define BLOCK_Q 64
#define PAGE_SIZE 16 // 也就是 BLOCK_KV

__global__ void paged_flash_attention_v1_kernel(
    __nv_bfloat16 *out,     // [num_seqs, max_q_len, num_heads, head_dim]
    const __nv_bfloat16 *q, // [num_seqs, max_q_len, num_heads, head_dim]
    const __nv_bfloat16
        *k_cache, // [num_blocks, num_kv_heads, PAGE_SIZE, head_dim]
    const __nv_bfloat16
        *v_cache,            // [num_blocks, num_kv_heads, PAGE_SIZE, head_dim]
    const int *block_tables, // [num_seqs, max_blocks_per_seq]
    const int *context_lens, // [num_seqs]
    const int q_len,         // 修改：因为当前调度的同一批次 prefill 的 token
                             // 数几乎一致，直接传纯标量即可
    int num_heads,
    int num_kv_heads,
    int max_blocks_per_seq,
    int head_dim,
    float scale) {
    // ==========================================
    // 1. 宏观坐标定位
    // ==========================================
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int q_chunk_idx = blockIdx.x; // 当前处理的是第几个 Q 分块

    int q_per_kv = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / q_per_kv;

    int context_len = context_lens[seq_idx];

    // 计算当前 Block 负责的 Q 的起始索引和真实有效长度
    int q_start_idx = q_chunk_idx * BLOCK_Q;
    if (q_start_idx >= q_len)
        return;

    // 如果最后一块不足 64 个，要做好边界保护
    int actual_q_len = min(BLOCK_Q, q_len - q_start_idx);

    // ==========================================
    // 2. 申请共享内存 (最存粹的 FlashAttention V1 显存复用魔法)
    // ==========================================
    // 刚好卡在 48KB 内的终极方案：32KB (sQ) + 8KB (sV) + 8KB (sK 和 sS 共用!) =
    // 48KB
    __shared__ float sQ[BLOCK_Q][128];
    __shared__ float sV[PAGE_SIZE][128];

    // 【魔法 1】因为 S (打分) 和 K (Key) 是交替使用的！我们算完内积后 K
    // 就不需要了！ 用 union 让它们复用同一片共享显存。
    // __shared__ 不能直接修饰匿名 union（nvcc 会静默忽略！warning
    // 必须给 union 一个名字，才能让 __shared__ 生效
    union KS_Union {
        float sK[PAGE_SIZE][128];
        float sS[BLOCK_Q]
                [PAGE_SIZE]; // BLOCK_Q=64, PAGE_SIZE=16. 64*16 = 1024 elements.
    };
    __shared__ KS_Union ks;

    // 【魔法 2】原来占用极其恐怖的 sO, sM, sL 统统不要放在共享内存里！
    // 因为这本来就是每个序列独享的数据，直接放到当前线程的“私人寄存器(Registers)”里！
    float my_m = -FLT_MAX;
    float my_l = 0.0f;
    float my_o[128]; // 虽然长达 128 维，但 NV 支持单线程最高分配 255 个寄存器！
    for (int i = 0; i < 128; i++) {
        my_o[i] = 0.0f;
    }

    // 协作加载Q
    int total_elements = actual_q_len * head_dim;
    for (int index = threadIdx.x; index < total_elements; index += blockDim.x) {
        int row = index / head_dim;
        int col = index % head_dim;
        long long q_offset =
            (long long)seq_idx * q_len * num_heads * head_dim +
            (long long)(q_start_idx + row) * num_heads * head_dim +
            (long long)head_idx * head_dim + col;
        sQ[row][col] = __bfloat162float(q[q_offset]);
    }
    __syncthreads();

    // 计算历史数据总共占用了多少个逻辑页 (向上取整)
    int num_logical_pages = (context_len + PAGE_SIZE - 1) / PAGE_SIZE;

    const int *my_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // 外层大循环：按页(Page)遍历历史 KV
    for (int logical_page = 0; logical_page < num_logical_pages;
         ++logical_page) {

        // 1. 查页表，获取物理块 ID
        int physical_block_id = my_block_table[logical_page];

        // 2. 计算这一页实际上有几个有效的 Token？
        // 绝大多数页都是满的 (PAGE_SIZE)，但最后一页可能没填满！
        int tokens_in_this_page =
            min(PAGE_SIZE, context_len - logical_page * PAGE_SIZE);

        // ==========================================
        // TODO: 协作加载 K 到 sK，加载 V 到 sV
        // ==========================================
        // 请模仿刚才加载 Q 的“一维打平搬运法”，
        // 用 128 个线程，把当前物理页里的 K 和 V 搬到 sK 和 sV 里。
        // （注意：K 和 V 的头部偏移量要用 kv_head_idx ！）

        // 补丁 2：显存载入必须清零防 NaN，同时用 long long 防越界！
        int kv_total_elements = PAGE_SIZE * head_dim; // 直接填满一整页！
        for (int i = threadIdx.x; i < kv_total_elements; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (row < tokens_in_this_page) {
                long long kv_offset = (long long)physical_block_id * PAGE_SIZE *
                                          num_kv_heads * head_dim +
                                      (long long)row * num_kv_heads * head_dim +
                                      (long long)kv_head_idx * head_dim + col;
                ks.sK[row][col] = __bfloat162float(k_cache[kv_offset]);
                sV[row][col] = __bfloat162float(v_cache[kv_offset]);
            } else {
                // 如果是没塞满的垃圾数据页，必须手动清零！否则后面 0.0 * NaN
                // 会变成 NaN 污染结果
                ks.sK[row][col] = 0.0f;
                sV[row][col] = 0.0f;
            }
        }

        __syncthreads();

        // ==========================================
        // 核心计算区 1：S = Q * K^T (SRAM 内部微型 GEMM)
        // ==========================================

        // 1. 线程的 2D 身份映射 (基于你的 4x2 架构！)
        // 128 个线程，要排成 16 行 8 列的阵型
        int tx = threadIdx.x % 8; // 列号: 0~7
        int ty = threadIdx.x / 8; // 行号: 0~15

        // 计算当前线程负责的 4x2 小矩阵，在 64x16 大矩阵中的起始坐标
        int s_row_start = ty * 4;
        int s_col_start = tx * 2;

        // 2. 申请私人寄存器
        float scores[4][2]; // 存放计算结果的 8 个累加器
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                scores[i][j] = 0.0f;
            }
        }
        float frag_q[4]; // 左手拿 4 个 Q
        float frag_k[2]; // 右手拿 2 个 K

        // 3. 沿着 head_dim (128) 维度做内积的展开 (外积魔法)
        for (int d = 0; d < head_dim; ++d) {

            // 步骤 A: 把 sQ 中对应的 4 个数读到 frag_q 寄存器中
            for (int i = 0; i < 4; ++i)
                frag_q[i] = sQ[s_row_start + i][d];

            // 步骤 B: 把 sK 中对应的 2 个数读到 frag_k 寄存器中
            for (int j = 0; j < 2; ++j)
                frag_k[j] = ks.sK[s_col_start + j][d];

            // 步骤 C: 4x2 的寄存器外积！
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 2; ++j) {
                    scores[i][j] += frag_q[i] * frag_k[j];
                }
            }
        }

        // ==========================================
        // 核心计算区 2：乘以 Scale 并应用 Causal Mask
        // ==========================================
        for (int i = 0; i < 4; ++i) {

            // 计算当前这个 Q 的全局绝对物理位置
            int q_absolute_pos =
                context_len - q_len + q_start_idx + s_row_start + i;

            for (int j = 0; j < 2; ++j) {
                // 乘以 scale (1.0 / sqrt(head_dim))
                scores[i][j] *= scale;

                // 计算当前这个 K 的全局绝对物理位置
                int k_absolute_pos = logical_page * PAGE_SIZE + s_col_start + j;

                int local_k_idx = s_col_start + j;
                // TODO: 极其优雅的 Causal Mask！
                // 如果 k_absolute_pos > q_absolute_pos，说明这是未来的词。
                if (local_k_idx >= tokens_in_this_page ||
                    k_absolute_pos > q_absolute_pos) {
                    scores[i][j] = -FLT_MAX;
                }
            }
        }

        // 必须等 128 个人全都算完了上面关于 sK 的乘法！！
        // 因为接下来我们要把结果洗回 sS，但它们共享同一片内存（被 Union 绑定）
        __syncthreads();

        // ==========================================
        // 步骤 1：把寄存器里的 8 个结果，统统放上公共工作台 sS
        // ==========================================
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) {
                ks.sS[s_row_start + i][s_col_start + j] = scores[i][j];
            }
        }
        __syncthreads(); // 必须等 128 个人全部放完！

        // ==========================================
        // 步骤 2：FlashAttention 核心状态更新 (Online Softmax & P*V)
        // ==========================================
        // 为了算法清晰，我们派前 64 个线程出列！
        // 每 1 个线程，独立负责 1 个 Q Token (也就是 sS 的 1 整行)
        if (threadIdx.x < BLOCK_Q) {
            int row = threadIdx.x; // 当前负责的 Q Token 编号 (0~63)

            // 1. 找出当前这新一页 (16个分数) 里的局部最大值 local_max
            float local_max = -FLT_MAX;
            for (int k = 0; k < PAGE_SIZE; ++k) {
                local_max = fmaxf(local_max, ks.sS[row][k]);
            }

            // 2. 华山论剑：历史最大值 vs 局部最大值，决出真正的全局新最大值！
            float m_new = fmaxf(my_m, local_max);

            // 3. 【天才之举】计算衰减因子 (Scaling Factor)
            // 必须把它们按比例缩小，缩小倍数就是 exp(旧最大值 - 新最大值)
            float diff = my_m - m_new;
            float exp_diff = (diff <= -1e20f) ? 0.0f : __expf(diff);

            // 4. 计算当前页的新指数和，并把 sS 的分数直接变成概率 P
            float local_sum = 0.0f;
            for (int k = 0; k < PAGE_SIZE; ++k) {
                float p = __expf(ks.sS[row][k] - m_new);
                // 极其关键的魔法修复：如果分数是 -FLT_MAX，减去它自己会变成
                // 0，exp(0) 会变成 1！ 这会导致所有被 Causal Mask 遮蔽掉的垃圾
                // token 全部获得 1.0 的超高概率，从而带来灾难级的污染！
                if (ks.sS[row][k] <= -1e20f) {
                    p = 0.0f;
                }
                ks.sS[row][k] = p; // 就地把分数覆写为概率！
                local_sum += p;
            }

            // 5. 更新全局指数和
            float l_new = my_l * exp_diff + local_sum;

            // 6. 更新终极输出向量 my_o！(P * V 融合)
            // 遍历 128 个维度
            for (int d = 0; d < head_dim; ++d) {
                // A. 历史输出要跟着衰减
                float o_val = my_o[d] * exp_diff;

                // B. 累加当前这一页 16 个 V 带来的新影响 (加权和)
                for (int k = 0; k < PAGE_SIZE; ++k) {
                    o_val += ks.sS[row][k] * sV[k][d];
                }

                // C. 写回我的私人寄存器
                my_o[d] = o_val;
            }

            // 7. 更新状态板，为下一个物理页的到来做准备
            my_m = m_new;
            my_l = l_new;
        }

        __syncthreads(); // 等待这 64 个人把状态更新完！
    }

    // 等待所有人把最后一页的状态更新完
    __syncthreads();

    // ==========================================
    // 最终章：归一化与全局写回 (写回 out 矩阵)
    // ==========================================
    // 我们再次召唤那 64 个工人，把大家算出存在个人寄存器里的
    // my_o，先倒回原来闲置的 sQ 共享工作台！
    if (threadIdx.x < BLOCK_Q) {
        int row = threadIdx.x;
        for (int d = 0; d < head_dim; ++d) {
            float final_val = my_l > 1e-10f ? my_o[d] / my_l : 0.0f; // 防 NaN
            sQ[row][d] = final_val; // 归一化后存进 sQ

            if (seq_idx == 0 && q_start_idx == 0) {
                if (isnan(final_val) || isinf(final_val)) {
                    printf(
                        "[NaN CATCH] log_pg=%d row=%d head=%d d=%d my_l=%f "
                        "my_o=%f\n",
                        num_logical_pages,
                        row,
                        head_idx,
                        d,
                        my_l,
                        my_o[d]);
                }
            }
        }
    }
    __syncthreads(); // 大家都倒完！

    // 最后，用这 128 个人（一维打平）全家出动，把 sQ
    // 全局写出，保证内存访问绝对合并！
    total_elements = actual_q_len * head_dim;

    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;

        long long out_idx =
            (long long)seq_idx * (q_len * num_heads * head_dim) +
            (long long)(q_start_idx + row) * (num_heads * head_dim) +
            (long long)head_idx * head_dim + col;

        out[out_idx] = __float2bfloat16(sQ[row][col]);
    }
}

// 统一的 Host 端启动接口
void launch_lucciola_attention(
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const int *block_tables,
    const int *context_lens,
    int num_seqs,
    int q_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    cudaStream_t stream) {

    // 我们目前专门为 Prefill(即 Chunk 大于 1) 提供 FlashAttention 加速
    if (q_len > 1) {
        // 算出在 Q 维度需要多少个 64 分块
        int blocks_x = (q_len + BLOCK_Q - 1) / BLOCK_Q;
        dim3 grid(blocks_x, num_seqs, num_heads);
        dim3 block(128); // 启动 4 个 Warp 协同工作

        paged_flash_attention_v1_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k_cache),
            reinterpret_cast<const __nv_bfloat16 *>(v_cache),
            block_tables,
            context_lens,
            q_len,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            head_dim,
            1.0f / sqrtf(head_dim));
    }
}

} // namespace lucciola::kernels