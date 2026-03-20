#include "kernels/learn/sgemm.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

namespace lucciola::kernels::learn {

void sgemm_cutlass_forward(
    float *C,
    const float *A,
    const float *B,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    using RowMajor = cutlass::layout::RowMajor;
    using Gemm = cutlass::gemm::device::
        Gemm<float, RowMajor, float, RowMajor, float, RowMajor, float>;

    Gemm gemm_op;

    // Construct arguments
    typename Gemm::Arguments args(
        {M, N, K}, // GemmCoord
        {A, K},    // A: pointer, stride
        {B, N},    // B: pointer, stride
        {C, N},    // C: pointer, stride
        {C, N},    // D: pointer, stride (output)
        {1.0f, 0.0f});

    size_t workspace_size = gemm_op.get_workspace_size(args);

    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t e = cudaMalloc(&workspace, workspace_size);
        if (e != cudaSuccess) {
            printf(
                "[CUTLASS] workspace alloc failed: %s\n",
                cudaGetErrorString(e));
            return;
        }
    }

    cutlass::Status status = gemm_op.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {
        status = gemm_op.run(stream);
    }

    if (workspace)
        cudaFree(workspace);
}

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

// 定义 Block 级别的分块 (整个 Block 负责计算 C 的 64x64 区域)
#define BM 64
#define BN 64
#define BK 8

// 定义 Thread 级别的分块 (每个线程负责计算 C 的 4x4 区域)
#define TM 4
#define TN 4

// 这是一个使用 2D 寄存器平铺的 GEMM
__global__ void sgemm_register_tiled_kernel(
    float *C, const float *A, const float *B, int M, int N, int K) {
    // 1. 计算线程负责的 C 矩阵的起始行和列
    // 因为一个 Block 处理 64x64，一个线程处理 4x4
    // 所以一个 Block 里只需要 (64/4) * (64/4) = 16 * 16 = 256 个线程
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y; // tx: 0~15, ty: 0~15

    // 当前线程负责的 4x4 区域在全局 C 矩阵中的左上角坐标
    int row_start = by * BM + ty * TM;
    int col_start = bx * BN + tx * TN;

    // 2. 申请共享内存
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // 3. 申请寄存器 (Thread Local)
    // accum 用来存放这 4x4=16 个计算结果
    float accum[TM][TN] = {0.0f};
    // frag_a 和 frag_b 用来缓存从共享内存读出来的 A 和 B
    float frag_a[TM];
    float frag_b[TN];

    // 计算当前线程在 Block 内部的 1D ID (用于协同搬运数据)
    int tid = ty * blockDim.x + tx; // 0 ~ 255

    // 外层循环：在 K 维度上滑动，每次滑动 BK (8) 步
    for (int k_step = 0; k_step < K; k_step += BK) {

        // =======================================================
        // 256 个线程协作将全局内存搬运到 sA 和 sB
        // BM*BK = 64*8 = 512 个元素。256 个线程，每个线程搬 2 个元素。
        // BN*BK = 64*8 = 512 个元素。同样每个线程搬 2 个。
        // =======================================================
        int a_idx = tid * 2; // 0, 2, 4 ... 510
        int a_row = a_idx / BK;
        int a_col = a_idx % BK;
        sA[a_row][a_col] = A[(by * BM + a_row) * K + (k_step + a_col)];
        sA[a_row][a_col + 1] = A[(by * BM + a_row) * K + (k_step + a_col + 1)];

        int b_idx = tid * 2;
        int b_row = b_idx / BN;
        int b_col = b_idx % BN;
        sB[b_row][b_col] = B[(k_step + b_row) * N + (bx * BN + b_col)];
        sB[b_row][b_col + 1] = B[(k_step + b_row) * N + (bx * BN + b_col + 1)];

        __syncthreads();

        // =======================================================
        // TODO 1: 核心寄存器计算引擎
        // =======================================================
        // 在这 8 步 (BK) 中，不断将 sA 和 sB 加载到寄存器，并做 4x4 的外积
        for (int k = 0; k < BK; ++k) {

            // 步骤 1: 将 sA 中对应的数据读入 frag_a 寄存器数组
            // 提示：当前线程负责行号是 ty * TM 到 ty * TM + 3。列号是 k。
            for (int i = 0; i < TM; ++i) {
                frag_a[i] = sA[ty * TM + i][k];
            }

            // 步骤 2: 将 sB 中对应的数据读入 frag_b 寄存器数组
            // 提示：当前线程负责列号是 tx * TN 到 tx * TN + 3。行号是 k。
            for (int j = 0; j < TN; ++j) {
                frag_b[j] = sB[k][tx * TN + j];
            }

            // 步骤 3: 寄存器级 4x4 矩阵乘法
            // 用两个嵌套的 for 循环 (i 从 0 到 3，j 从 0 到 3)
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += frag_a[i] * frag_b[j];
                }
            }
        }

        __syncthreads();
    }

    // =======================================================
    // TODO 2: 将这 16 个结果写回全局显存 C
    // =======================================================
    // 遍历 TM 和 TN (4x4)
    // 注意越界保护：如果 (row_start + i) < M 且 (col_start + j) < N
    // 将 accum[i][j] 写入 C[(row_start + i) * N + (col_start + j)]
    for (int i = 0; i < TM; ++i) {
        int global_row = row_start + i;
        for (int j = 0; j < TN; ++j) {
            int global_col = col_start + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accum[i][j];
            }
        }
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

void sgemm_register_tiled_forward(
    float *C,
    const float *A,
    const float *B,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_register_tiled_kernel<<<grid, block, 0, stream>>>(C, A, B, M, N, K);
}

} // namespace lucciola::kernels::learn