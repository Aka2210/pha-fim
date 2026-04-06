#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include "pha_fim_gpu.hpp"

using namespace std;

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    cerr << "CUDA error: " << cudaGetErrorString(e) \
          << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    exit(1); \
  } \
} while(0)

DevicePackedDataset upload_packed_dataset_to_gpu(const PackedDataset& ds) {
  DevicePackedDataset dds;
  dds.items_count = static_cast<int>(ds.items_flat.size());
  dds.tx_count = static_cast<int>(ds.start.size());

  if (!ds.items_flat.empty()) {
    CUDA_CHECK(cudaMalloc(&dds.d_items, ds.items_flat.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dds.d_items,
                          ds.items_flat.data(),
                          ds.items_flat.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  if (!ds.start.empty()) {
    CUDA_CHECK(cudaMalloc(&dds.d_start, ds.start.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dds.d_start,
                          ds.start.data(),
                          ds.start.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  if (!ds.tx_len.empty()) {
    CUDA_CHECK(cudaMalloc(&dds.d_len, ds.tx_len.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dds.d_len,
                          ds.tx_len.data(),
                          ds.tx_len.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  return dds;
}

void free_device_packed_dataset(DevicePackedDataset& dds) {
  if (dds.d_items) {
    cudaFree(dds.d_items);
    dds.d_items = nullptr;
  }
  if (dds.d_start) {
    cudaFree(dds.d_start);
    dds.d_start = nullptr;
  }
  if (dds.d_len) {
    cudaFree(dds.d_len);
    dds.d_len = nullptr;
  }

  dds.items_count = 0;
  dds.tx_count = 0;
}

__global__ void init_population_kernel(
    uint8_t* pop, int P, int m, unsigned int seed
) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= P) return;

  unsigned int x = pid ^ seed;

  for (int j = 0; j < m; ++j) {
    x = x * 1103515245u + 12345u;
    pop[pid * m + j] = (x % 2);
  }

  bool any = false;
  for (int j = 0; j < m; ++j) {
    if (pop[pid * m + j]) {
        any = true;
        break;
    }
  }

  if (!any && m > 0) {
    x = x * 1103515245u + 12345u;
    int pos = x % m;
    pop[pid * m + pos] = 1;
  }
}

void run_gpu_fim(
  const PackedDataset& ds1,
  const PackedDataset& ds2,
  const PackedDataset& ds3,
  int N, int m, int min_sup_count,
  int P, int iters
) {
  // upload three CPU-preprocessed datasets to GPU
  DevicePackedDataset dds1 = upload_packed_dataset_to_gpu(ds1);
  DevicePackedDataset dds2 = upload_packed_dataset_to_gpu(ds2);
  DevicePackedDataset dds3 = upload_packed_dataset_to_gpu(ds3);

  // allocate population-related device memory
  uint8_t* d_pop = nullptr;
  uint8_t* d_pop1 = nullptr;
  uint8_t* d_pop2 = nullptr;
  uint8_t* d_pop3 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_pop,  P * m * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_pop1, P * m * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_pop2, P * m * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_pop3, P * m * sizeof(uint8_t)));

  // initialize base population on GPU
  int block = 256;
  int grid = (P + block - 1) / block;
  init_population_kernel<<<grid, block>>>(d_pop, P, m, 42);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // main iteration
  for (int it = 0; it < iters; ++it) 
  {
    CUDA_CHECK(cudaMemcpy(d_pop1, d_pop, P * m * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_pop2, d_pop, P * m * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_pop3, d_pop, P * m * sizeof(uint8_t), cudaMemcpyDeviceToDevice));

    // TODO:
    // search on d_pop1 / d_pop2 / d_pop3
    // fitness using dds1 / dds2 / dds3
    // merge back into d_pop
  }

  // cleanup dataset device memory
  free_device_packed_dataset(dds1);
  free_device_packed_dataset(dds2);
  free_device_packed_dataset(dds3);

  if (d_pop)  cudaFree(d_pop);
  if (d_pop1) cudaFree(d_pop1);
  if (d_pop2) cudaFree(d_pop2);
  if (d_pop3) cudaFree(d_pop3);
}