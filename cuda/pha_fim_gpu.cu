#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <random>
#include <algorithm>
#include "pha_fim_gpu.hpp"

using namespace std;

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    cerr << "CUDA error: " << cudaGetErrorString(e) << "\n"; \
    exit(1); \
  } \
} while(0)

void run_gpu_fim(
    const vector<int>& items_flat,
    const vector<int>& start,
    const vector<int>& len,
    int N, int m, int min_sup_count,
    int P, int iters
) {
    // 1) copy dataset to GPU
    int *d_items=nullptr, *d_start=nullptr, *d_len=nullptr;
    CUDA_CHECK(cudaMalloc(&d_items, items_flat.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_start, start.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_len,   len.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_items, items_flat.data(), items_flat.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start, start.data(),      start.size()*sizeof(int),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len,   len.data(),        len.size()*sizeof(int),        cudaMemcpyHostToDevice));

    // cleanup
    cudaFree(d_items); cudaFree(d_start); cudaFree(d_len);
}