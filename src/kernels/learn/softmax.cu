#include "kernels/learn/softmax.h"
#include <cfloat> // 用于获取单精度浮点数的极小值 -FLT_MAX

namespace lucciola::kernels::learn {

// 1. CUDA 核函数：计算 1D Softmax
__global__ void naive_softmax_kernel(float *scores, int valid_len) {
    // 为了最简单的学习，我们先假设只启动了 1 个线程（单核运行）
    // 也就是说，这个函数就相当于跑在一个 CPU 核心上一样

    // ==========================================
    // TODO 1: 遍历 0 到 valid_len - 1，找出 scores 里的最大值 (max_val)
    // 提示：初始化 max_val 为 -FLT_MAX
    // ==========================================
    float max_val = -FLT_MAX;

    for (int index = 0; index < valid_len; ++index) {
        max_val = fmaxf(max_val, scores[index]);
    }

    // ==========================================
    // TODO 2: 遍历 0 到 valid_len - 1，计算 e^(x_i - max_val) 的总和 (sum)
    // 提示：可以使用 CUDA 内置的数学函数 expf()
    // ==========================================
    float sum = 0.0f;
    for (int index = 0; index < valid_len; ++index) {
        sum += __expf(scores[index] - max_val);
    }

    // ==========================================
    // TODO 3: 遍历 0 到 valid_len - 1，计算最终的概率并写回 scores 数组
    // 提示：公式为 scores[i] = expf(scores[i] - max_val) / sum
    // ==========================================
    for (int index = 0; index < valid_len; ++index) {
        scores[index] = __expf(scores[index] - max_val) / sum;
    }
}

__global__ void online_softmax_kernel(float *scores, int valid_len) {
    // 依然假设单线程运行

    float max_val = -FLT_MAX; // 对应公式里的 m
    float sum = 0.0f;         // 对应公式里的 d

    // ==========================================
    // TODO 1: 仅用 1 个 for 循环（一趟扫描），同时更新 max_val 和 sum
    // 提示：
    // 1. 临时保存旧的 max_val (比如叫 old_max)
    // 2. 更新出新的 max_val
    // 3. 用魔法公式更新 sum: sum = sum * exp(old_max - max_val) + exp(当前值 -
    // max_val)
    // ==========================================
    for (int index = 0; index < valid_len; ++index) {
        float old_max = max_val;
        max_val = fmaxf(max_val, scores[index]);
        sum = sum * __expf(old_max - max_val) + __expf(scores[index] - max_val);
    }

    // ==========================================
    // TODO 2: 再用 1 个 for 循环，计算最终概率并写回
    // ==========================================
    for (int index = 0; index < valid_len; ++index) {
        scores[index] = __expf(scores[index] - max_val) / sum;
    }
}

// 常量：所有 32 个线程都参与
#define FULL_MASK 0xffffffff

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

__global__ void warp_softmax_kernel(float *scores, int valid_len) {
    // 获取当前线程在 Warp 中的编号 (0 到 31)
    int tid = threadIdx.x;

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    // ==========================================
    // TODO 1: 跨步循环 (Stride Loop)
    // 每个线程只处理属于自己的那些元素 (i = tid; i < valid_len; i += 32)
    // 在循环内部，用你之前写的 Online Softmax 逻辑更新 thread_max 和 thread_sum
    // ==========================================
    for (int index = tid; index < valid_len; index += blockDim.x) {
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, scores[index]);
        thread_sum = thread_sum * __expf(old_max - thread_max) +
                     __expf(scores[index] - thread_max);
    }

    // ==========================================
    // TODO 2: Warp 级规约与对齐 (Warp Reduction)
    // 1. 算出 global_max
    // 2. 对齐本线程的 thread_sum
    // 3. 算出 global_sum
    // ==========================================
    float global_max = warpReduceMax(thread_max);
    // global_max = __shfl_up_sync(FULL_MASK, global_max, tid);
    global_max = __shfl_sync(FULL_MASK, global_max, 0);
    thread_sum = thread_sum * __expf(thread_max - global_max);

    float global_sum = warpReduceSum(thread_sum);
    // global_sum = __shfl_up_sync(FULL_MASK, global_sum, tid);
    global_sum = __shfl_sync(FULL_MASK, global_sum, 0);

    // ==========================================
    // TODO 3: 再次跨步循环，计算概率并写回
    // ==========================================

    for (int index = tid; index < valid_len; index += blockDim.x) {
        scores[index] = __expf(scores[index] - global_max) / global_sum;
    }
}

// 2. 宿主端包装器
void naive_softmax_forward(float *scores, int valid_len, cudaStream_t stream) {
    // 启动 1 个 Block，每个 Block 只有 1 个 Thread (最慢但最安全的版本)
    dim3 grid(1);
    dim3 block(1);

    naive_softmax_kernel<<<grid, block, 0, stream>>>(scores, valid_len);
}

void online_softmax_forward(float *scores, int valid_len, cudaStream_t stream) {
    dim3 grid(1);
    dim3 block(1);

    online_softmax_kernel<<<grid, block, 0, stream>>>(scores, valid_len);
}

void warp_softmax_forward(float *scores, int valid_len, cudaStream_t stream) {
    dim3 grid(1);
    dim3 block(32);

    warp_softmax_kernel<<<grid, block, 0, stream>>>(scores, valid_len);
}

} // namespace lucciola::kernels::learn