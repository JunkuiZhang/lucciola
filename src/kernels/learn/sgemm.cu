#include "kernels/learn/sgemm.h"

namespace lucciola::kernels::learn {

const int BLOCK_SIZE = 16;

__global__ void sgemm_naive_kernel(
    float *C, const float *A, const float *B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// A: [M, K], B: [K, N], C: [M, N]
__global__ void sgemm_tiled_kernel(
    float *__restrict__ C,
    const float *__restrict__ A,
    const float *__restrict__ B,
    int M,
    int N,
    int K) {
    // ==========================================
    // TODO 1: 确定当前线程在输出矩阵 C 中的全局坐标
    // ==========================================
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 确定当前线程在当前 Block 内部的局部坐标 (0 ~ 15)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 申请两块共享内存，用来存放 A 和 B 的 16x16 小块
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // 用来累加当前线程负责的那个 C 元素的值
    float sum = 0.0f;

    // ==========================================
    // TODO 2: 外层循环，让 16x16 的滑块在 K 维度上滑动
    // ==========================================
    // 每次滑动 BLOCK_SIZE (16) 步
    for (int k_step = 0; k_step < K; k_step += BLOCK_SIZE) {

        // ==========================================
        // TODO 3: 协作搬运数据到共享内存
        // ==========================================
        // 当前线程 (ty, tx) 负责把 A 和 B 中的 1 个元素搬到 sA 和 sB 中。
        // 注意越界保护：如果行或列超出了矩阵的真实边界，就填入 0.0f (Padding)

        sA[ty][tx] =
            (row < M && k_step + tx < K) ? A[row * K + (k_step + tx)] : 0.0f;
        sB[ty][tx] =
            (k_step + ty < K && col < N) ? B[(k_step + ty) * N + col] : 0.0f;

        // ==========================================
        // TODO 4: 同步屏障
        // ==========================================
        // 必须等 Block 里的所有人都把数据搬完，才能开始计算！
        // 调用 CUDA 的同步指令：
        __syncthreads();

        // ==========================================
        // TODO 5: 内层循环，做 16 次乘加运算
        // ==========================================
        // 在共享内存中计算点积：将 sA 的第 ty 行，乘以 sB 的第 tx 列
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }

        // ==========================================
        // TODO 6: 再次同步屏障
        // ==========================================
        // 必须等所有人都算完这轮的乘法，才能进入下一轮 k_step
        // 去覆盖共享内存里的旧数据！
        __syncthreads();
    }

    // 写入结果 C (注意越界保护)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 宿主端调用
void sgemm_naive_forward(
    float *C,
    const float *A,
    const float *B,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemm_naive_kernel<<<grid, block, 0, stream>>>(C, A, B, M, N, K);
}

void sgemm_tiled_forward(
    float *C,
    const float *A,
    const float *B,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemm_tiled_kernel<<<grid, block, 0, stream>>>(C, A, B, M, N, K);
}

} // namespace lucciola::kernels::learn