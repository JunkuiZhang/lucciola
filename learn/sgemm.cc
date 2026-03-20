#include "kernels/learn/sgemm.h"
#include <cmath>
#include <format>

#define BLOCK_SIZE 16

// ========================================================
// 3. CPU 验证函数
// ========================================================
bool verify_matrix(float *C_cpu, float *C_gpu, int M, int N) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-4) {
            std::format(
                "Mismatch at {}: CPU={}, GPU={}", i, C_cpu[i], C_gpu[i]);
            return false;
        }
    }
    return true;
}

// ========================================================
// 主函数 Benchmarking
// ========================================================
int main() {
    // 矩阵维度 (1024x1024 是个不错的测试大小)
    int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // CPU 内存分配
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_cpu = (float *)malloc(size_C);
    float *h_C_gpu = (float *)malloc(size_C);

    // 随机初始化
    for (int i = 0; i < M * K; i++)
        h_A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++)
        h_B[i] = (rand() % 100) / 100.0f;

    // GPU 显存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 网格与线程块设置
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    // 计算浮点运算次数 (M * N 个元素，每个元素 K 次乘法 + K 次加法)
    double flops = 2.0 * M * N * K;

    // ---------------------------------------------------------
    // 测试 1: CPU 参考结果 (很慢，需要等几秒钟)
    // ---------------------------------------------------------
    printf("Computing CPU reference...\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k)
                sum += h_A[i * K + k] * h_B[k * N + j];
            h_C_cpu[i * N + j] = sum;
        }
    }

    // ---------------------------------------------------------
    // 测试 2: GPU Naive 版本
    // ---------------------------------------------------------
    cudaMemset(d_C, 0, size_C);
    cudaEventRecord(start);
    lucciola::kernels::learn::sgemm_naive_forward(d_C, d_A, d_B, M, N, K, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    bool naive_pass = verify_matrix(h_C_cpu, h_C_gpu, M, N);
    printf(
        "Naive Kernel  | Correct: %s | Time: %8.3f ms | Performance: %8.2f "
        "GFLOPS\n",
        naive_pass ? "YES" : "NO",
        ms,
        (flops / 1e9) / (ms / 1000.0f));

    // ---------------------------------------------------------
    // 测试 3: GPU Tiled 版本 (你写的)
    // ---------------------------------------------------------
    cudaMemset(d_C, 0, size_C);
    cudaEventRecord(start);
    lucciola::kernels::learn::sgemm_tiled_forward(d_C, d_A, d_B, M, N, K, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    bool tiled_pass = verify_matrix(h_C_cpu, h_C_gpu, M, N);
    printf(
        "Tiled Kernel  | Correct: %s | Time: %8.3f ms | Performance: %8.2f "
        "GFLOPS\n",
        tiled_pass ? "YES" : "NO",
        ms,
        (flops / 1e9) / (ms / 1000.0f));

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    return 0;
}