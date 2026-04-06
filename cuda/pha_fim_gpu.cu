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

    // --------------------------------------------------
    // TODO: allocate population-related device memory
    // --------------------------------------------------

    // --------------------------------------------------
    // TODO: initialize base population on GPU
    // --------------------------------------------------

    // --------------------------------------------------
    // TODO: main iteration

    // --------------------------------------------------
    // cleanup dataset device memory
    // --------------------------------------------------
    free_device_packed_dataset(dds1);
    free_device_packed_dataset(dds2);
    free_device_packed_dataset(dds3);
}