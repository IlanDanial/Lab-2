#include <cuda_runtime.h>
#include <cmath> 
#include <cstdio>
#include <cstdlib> 
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
using namespace std;

inline void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
#define CUDA_CHECK(call) cuda_check(call)

__global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
            C[row * N + col] = sum;
            }
}

bool close_enough(float got, float ref) {
if (!std::isfinite(got)) return false;
float tolerance = 1.0e-3f + 1.0e-3f * std::fabs(ref);
return std::fabs(got - ref) <= tolerance;
}




int main(int argc, char** argv) {
    //
    int M = (argc > 1) ? std::atoi(argv[1]) : 10;
    int K = (argc > 2) ? std::atoi(argv[2]) : 10;
    int N = (argc > 3) ? std::atoi(argv[3]) : 10;
    float *A_d, *B_d, *C_d;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    std::vector<float> A_h(M * K), B_h(K * N);
    std::vector<float> C_H(M * N), C_cpu(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < A_h.size(); ++i) A_h[i] = dist(gen);
    for (size_t i = 0; i < B_h.size(); ++i) B_h[i] = dist(gen);

    matmul_cpu(A_h.data(), B_h.data(), C_cpu.data(), M, K,N);

    CUDA_CHECK(cudaMalloc(&A_d, bytesA));
    CUDA_CHECK(cudaMalloc(&B_d, bytesB));
    CUDA_CHECK(cudaMalloc(&C_d, bytesC));


    // Copy A and B
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), bytesB, cudaMemcpyHostToDevice));
    // launch matmul
    dim3 block(16,16);
    dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);

    matmul<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
    // check launch
    CUDA_CHECK(cudaGetLastError());
    // synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
    //copy C back
    CUDA_CHECK(cudaMemcpy(C_H.data(), C_d, bytesC, cudaMemcpyDeviceToHost));
    // , compare,
    bool pass = true;
    for (size_t i = 0; i < C_H.size(); i++){
    pass = close_enough(C_H[i], C_cpu[i]);
    if (pass == false){
        cout << " GPU: " << C_H[i] << " CPU: " << C_cpu[i] << " index: " << i << endl;
        break;
    }
    }
    //  then free.
    CUDA_CHECK(cudaFree(A_d)); CUDA_CHECK(cudaFree(B_d)); CUDA_CHECK(cudaFree(C_d));
    if (pass == false){
        cout << "Fail" << "M: " << M << "K: " << K << "N: " << N << endl;
    }
    else{
        cout << "Pass" << "M: " << M << " K: " << K << " N: " << N << endl;
    }

    vector<int> blocks = {16, 32};

    CUDA_CHECK(cudaMalloc(&A_d, bytesA));
    CUDA_CHECK(cudaMalloc(&B_d, bytesB));
    CUDA_CHECK(cudaMalloc(&C_d, bytesC));
    // Copy A and B
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), bytesB, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    vector<vector<float>> times(blocks.size());
    

    for (size_t i = 0; i < blocks.size(); i++){
    // launch matmul
    dim3 block1(blocks[i],256/blocks[i]);
    dim3 grid1((N+block1.x-1)/block1.x, (M+block1.y-1)/block1.y);
        for (int rep = 0; rep < 10; rep++){
            CUDA_CHECK(cudaEventRecord(start));
            matmul<<<grid1, block1>>>(A_d, B_d, C_d, M, K, N);
            CUDA_CHECK(cudaEventRecord(stop));
            // check launch
            CUDA_CHECK(cudaGetLastError());
            // synchronize
            CUDA_CHECK(cudaDeviceSynchronize());
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            times[i].push_back(ms);
        }
    }
    CUDA_CHECK(cudaFree(A_d)); CUDA_CHECK(cudaFree(B_d)); CUDA_CHECK(cudaFree(C_d));

    // median and output
    for (size_t i = 0; i < blocks.size(); i++){
        sort(times[i].begin(), times[i].end());
        float median = times[i][times[i].size() / 2];
        cout << "block " << blocks[i] << "x" << 256/blocks[i]
             << " M: " << M << " K: " << K << " N: " << N
             << " median kernel time: " << median << " ms" << endl;
    }

    return pass ? 0 : 1;
}