#include "kernels/learn/softmax.h"
#include <iostream>

int main() {
    std::cout << "Learn Softmax Kernel" << std::endl;
    const int num_tokens = 4;

    float h_scores[num_tokens] = {1.0f, 2.0f, 3.0f, 4.0f};
    float *d_scores;
    cudaMalloc(&d_scores, num_tokens * sizeof(float));
    cudaMemcpy(
        d_scores, h_scores, num_tokens * sizeof(float), cudaMemcpyHostToDevice);
    lucciola::kernels::learn::naive_softmax_forward(d_scores, num_tokens, 0);
    cudaMemcpy(
        h_scores, d_scores, num_tokens * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Naive Softmax Scores: ";
    for (int i = 0; i < num_tokens; i++) {
        std::cout << h_scores[i] << " ";
    }
    std::cout << std::endl;

    float new_scores[num_tokens] = {1.0f, 2.0f, 3.0f, 4.0f};
    cudaMemcpy(
        d_scores,
        new_scores,
        num_tokens * sizeof(float),
        cudaMemcpyHostToDevice);
    lucciola::kernels::learn::online_softmax_forward(d_scores, num_tokens, 0);
    cudaMemcpy(
        new_scores,
        d_scores,
        num_tokens * sizeof(float),
        cudaMemcpyDeviceToHost);
    std::cout << "Online Softmax Scores: ";
    for (int i = 0; i < num_tokens; i++) {
        std::cout << new_scores[i] << " ";
    }
    std::cout << std::endl;
    cudaFree(d_scores);

    const int long_num_tokens = 1024;
    float h_scores_1[long_num_tokens] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int index = 0; index < long_num_tokens; ++index) {
        h_scores_1[index] = (float)(index + 1);
    }

    float *long_d_scores;
    cudaMalloc(&long_d_scores, long_num_tokens * sizeof(float));

    cudaMemcpy(
        long_d_scores,
        h_scores_1,
        long_num_tokens * sizeof(float),
        cudaMemcpyHostToDevice);
    lucciola::kernels::learn::warp_softmax_forward(
        long_d_scores, long_num_tokens, 0);
    cudaMemcpy(
        h_scores_1,
        long_d_scores,
        long_num_tokens * sizeof(float),
        cudaMemcpyDeviceToHost);
    std::cout << "Warp Softmax Scores: ";
    for (int i = 0; i < long_num_tokens; i++) {
        std::cout << h_scores_1[i] << " ";
    }
    std::cout << std::endl;

    for (int index = 0; index < long_num_tokens; ++index) {
        h_scores_1[index] = (float)(index + 1);
    }
    cudaMemcpy(
        long_d_scores,
        h_scores_1,
        long_num_tokens * sizeof(float),
        cudaMemcpyHostToDevice);
    lucciola::kernels::learn::batched_warp_softmax_forward(
        long_d_scores, long_num_tokens, 1, 0);
    cudaMemcpy(
        h_scores_1,
        long_d_scores,
        long_num_tokens * sizeof(float),
        cudaMemcpyDeviceToHost);
    std::cout << "Batched Warp Softmax Scores: ";
    for (int i = 0; i < long_num_tokens; i++) {
        std::cout << h_scores_1[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(long_d_scores);
    return 0;
}