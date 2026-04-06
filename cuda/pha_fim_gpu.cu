#include "pha_fim_gpu.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__    \
           << ":" << __LINE__ << "\n";                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr float ALPHA = 0.1f;
static constexpr float BETA = 0.3f;
static constexpr float GAMMA = 0.6f;

DevicePackedDataset upload_packed_dataset_to_gpu(const PackedDataset &ds) {
  DevicePackedDataset dds;
  dds.items_count = static_cast<int>(ds.items_flat.size());
  dds.tx_count = static_cast<int>(ds.start.size());

  if (!ds.items_flat.empty()) {
    CUDA_CHECK(cudaMalloc(&dds.d_items, ds.items_flat.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dds.d_items, ds.items_flat.data(),
                          ds.items_flat.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  if (!ds.start.empty()) {
    CUDA_CHECK(cudaMalloc(&dds.d_start, ds.start.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dds.d_start, ds.start.data(),
                          ds.start.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  if (!ds.tx_len.empty()) {
    CUDA_CHECK(cudaMalloc(&dds.d_len, ds.tx_len.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dds.d_len, ds.tx_len.data(),
                          ds.tx_len.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  return dds;
}

void free_device_packed_dataset(DevicePackedDataset &dds) {
  if (dds.d_items)
    cudaFree(dds.d_items);
  if (dds.d_start)
    cudaFree(dds.d_start);
  if (dds.d_len)
    cudaFree(dds.d_len);

  dds.d_items = nullptr;
  dds.d_start = nullptr;
  dds.d_len = nullptr;
  dds.items_count = 0;
  dds.tx_count = 0;
}

__device__ __forceinline__ unsigned int lcg_next(unsigned int x) {
  return x * 1103515245u + 12345u;
}

// one thread = one individual
__global__ void init_population_kernel(uint8_t *pop, int *pop_count, int P,
                                       int m, unsigned int seed) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= P)
    return;

  unsigned int x = pid ^ seed;
  int count = 0;

  for (int j = 0; j < m; ++j) {
    x = lcg_next(x);
    uint8_t bit = static_cast<uint8_t>(x & 1u);
    pop[pid * m + j] = bit;
    count += bit;
  }

  if (count == 0 && m > 0) {
    x = lcg_next(x);
    int pos = static_cast<int>(x % m);
    pop[pid * m + pos] = 1;
    count = 1;
  }

  pop_count[pid] = count;
}

// one thread = one individual
// three-view search:
// view 1 -> search front segment of sm1
// view 2 -> search front segment of sm2
// view 3 -> search tail segment of sm3
__global__ void search_kernel(uint8_t *pop, int *pop_count, int P, int m,
                              float alpha, float beta, float gamma,
                              int view_id, // 1 = sm1, 2 = sm2, 3 = sm3
                              unsigned int seed, int iter) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= P || m <= 0)
    return;

  float ratio;
  if (pid < P / 3) {
    ratio = alpha;
  } else if (pid < 2 * P / 3) {
    ratio = beta;
  } else {
    ratio = gamma;
  }

  int span = max(1, static_cast<int>(ratio * m));
  if (span > m)
    span = m;

  unsigned int x = (pid ^ seed ^ (iter * 2654435761u));
  x = lcg_next(x);

  int pos = 0;

  if (view_id == 1) {
    pos = static_cast<int>(x % span);
  } else if (view_id == 2) {
    pos = static_cast<int>(x % span);
  } else {
    pos = m - span + static_cast<int>(x % span);
    if (pos >= m)
      pos = m - 1;
  }

  int idx = pid * m + pos;

  if (pop[idx]) {
    pop[idx] = 0;
    pop_count[pid] -= 1;

    // avoid empty itemset
    if (pop_count[pid] == 0) {
      x = lcg_next(x);
      int pos2 = static_cast<int>(x % m);
      pop[pid * m + pos2] = 1;
      pop_count[pid] = 1;
    }
  } else {
    pop[idx] = 1;
    pop_count[pid] += 1;
  }
}

// grid.y = P, grid.x covers transactions
// one thread handles one (individual, transaction)
__global__ void support_fitness_kernel(const uint8_t *pop, const int *pop_count,
                                       int *fitness, const int *items,
                                       const int *start, const int *len, int N,
                                       int m) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int pid = blockIdx.y;

  if (tx >= N)
    return;

  int target = pop_count[pid];
  if (target <= 0)
    return;

  int found = 0;
  int base = start[tx];
  int L = len[tx];

  for (int k = 0; k < L; ++k) {
    int item = items[base + k];
    if (item >= 0 && item < m && pop[pid * m + item]) {
      found++;
    }
  }

  if (found == target) {
    atomicAdd(&fitness[pid], 1);
  }
}

// one thread = one individual
// choose best among base / pop1 / pop2 / pop3 by fitness
__global__ void
ring_topology_kernel(uint8_t *base_pop, int *base_count, int *base_fit,

                     const uint8_t *pop1, const int *count1, const int *fit1,

                     const uint8_t *pop2, const int *count2, const int *fit2,

                     const uint8_t *pop3, const int *count3, const int *fit3,

                     int P, int m) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= P)
    return;

  int left = (pid == 0) ? (P - 1) : (pid - 1);
  int self = pid;
  int right = (pid == P - 1) ? 0 : (pid + 1);

  int best_view = 1;
  int best_pid = left;
  int best_fit_val = fit1[left];

  auto try_candidate = [&](int view, int cand_pid, int cand_fit) {
    if (cand_fit > best_fit_val) {
      best_fit_val = cand_fit;
      best_view = view;
      best_pid = cand_pid;
    }
  };

  // view 1 neighbors
  try_candidate(1, self, fit1[self]);
  try_candidate(1, right, fit1[right]);

  // view 2 neighbors
  try_candidate(2, left, fit2[left]);
  try_candidate(2, self, fit2[self]);
  try_candidate(2, right, fit2[right]);

  // view 3 neighbors
  try_candidate(3, left, fit3[left]);
  try_candidate(3, self, fit3[self]);
  try_candidate(3, right, fit3[right]);

  const uint8_t *src_pop = nullptr;
  const int *src_count = nullptr;
  const int *src_fit = nullptr;

  if (best_view == 1) {
    src_pop = pop1;
    src_count = count1;
    src_fit = fit1;
  } else if (best_view == 2) {
    src_pop = pop2;
    src_count = count2;
    src_fit = fit2;
  } else {
    src_pop = pop3;
    src_count = count3;
    src_fit = fit3;
  }

  for (int j = 0; j < m; ++j) {
    base_pop[pid * m + j] = src_pop[best_pid * m + j];
  }
  base_count[pid] = src_count[best_pid];
  base_fit[pid] = src_fit[best_pid];
}

void run_gpu_fim(const PackedDataset &ds1, const PackedDataset &ds2,
                 const PackedDataset &ds3, int N, int m, int min_sup_count,
                 int P, int iters) {
  DevicePackedDataset dds1 = upload_packed_dataset_to_gpu(ds1);
  DevicePackedDataset dds2 = upload_packed_dataset_to_gpu(ds2);
  DevicePackedDataset dds3 = upload_packed_dataset_to_gpu(ds3);

  uint8_t *d_pop = nullptr, *d_pop1 = nullptr, *d_pop2 = nullptr,
          *d_pop3 = nullptr;
  int *d_count = nullptr, *d_count1 = nullptr, *d_count2 = nullptr,
      *d_count3 = nullptr;
  int *d_fit = nullptr, *d_fit1 = nullptr, *d_fit2 = nullptr, *d_fit3 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_pop, P * m * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_pop1, P * m * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_pop2, P * m * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_pop3, P * m * sizeof(uint8_t)));

  CUDA_CHECK(cudaMalloc(&d_count, P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count1, P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count2, P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count3, P * sizeof(int)));

  CUDA_CHECK(cudaMalloc(&d_fit, P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fit1, P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fit2, P * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fit3, P * sizeof(int)));

  int pop_block = 256;
  int pop_grid = (P + pop_block - 1) / pop_block;

  init_population_kernel<<<pop_grid, pop_block>>>(d_pop, d_count, P, m, 42u);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemset(d_fit, 0, P * sizeof(int)));

  int fit_block = 256;
  int fit_grid_x = (N + fit_block - 1) / fit_block;
  dim3 fit_grid(fit_grid_x, P);

  // initial fitness for base population on ds1
  support_fitness_kernel<<<fit_grid, fit_block>>>(
      d_pop, d_count, d_fit, dds1.d_items, dds1.d_start, dds1.d_len, N, m);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int it = 0; it < iters; ++it) {
    CUDA_CHECK(cudaMemcpy(d_pop1, d_pop, P * m * sizeof(uint8_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_pop2, d_pop, P * m * sizeof(uint8_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_pop3, d_pop, P * m * sizeof(uint8_t),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(d_count1, d_count, P * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_count2, d_count, P * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_count3, d_count, P * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemset(d_fit1, 0, P * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_fit2, 0, P * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_fit3, 0, P * sizeof(int)));

    // three-view search
    search_kernel<<<pop_grid, pop_block>>>(d_pop1, d_count1, P, m, ALPHA, BETA,
                                           GAMMA, 1, 101u, it);
    search_kernel<<<pop_grid, pop_block>>>(d_pop2, d_count2, P, m, ALPHA, BETA,
                                           GAMMA, 2, 202u, it);
    search_kernel<<<pop_grid, pop_block>>>(d_pop3, d_count3, P, m, ALPHA, BETA,
                                           GAMMA, 3, 303u, it);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // fitness step
    support_fitness_kernel<<<fit_grid, fit_block>>>(
        d_pop1, d_count1, d_fit1, dds1.d_items, dds1.d_start, dds1.d_len, N, m);
    support_fitness_kernel<<<fit_grid, fit_block>>>(
        d_pop2, d_count2, d_fit2, dds2.d_items, dds2.d_start, dds2.d_len, N, m);
    support_fitness_kernel<<<fit_grid, fit_block>>>(
        d_pop3, d_count3, d_fit3, dds3.d_items, dds3.d_start, dds3.d_len, N, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // merge/select step
    ring_topology_kernel<<<pop_grid, pop_block>>>(
        d_pop, d_count, d_fit, d_pop1, d_count1, d_fit1, d_pop2, d_count2,
        d_fit2, d_pop3, d_count3, d_fit3, P, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  vector<int> h_fit(P, 0);
  CUDA_CHECK(
      cudaMemcpy(h_fit.data(), d_fit, P * sizeof(int), cudaMemcpyDeviceToHost));

  int best = 0;
  int qualified = 0;
  for (int v : h_fit) {
    best = max(best, v);
    if (v >= min_sup_count)
      qualified++;
  }

  cout << "[GPU FIM] best support found = " << best
       << ", min_sup_count = " << min_sup_count
       << ", qualified individuals = " << qualified << "\n";

  free_device_packed_dataset(dds1);
  free_device_packed_dataset(dds2);
  free_device_packed_dataset(dds3);

  if (d_pop)
    cudaFree(d_pop);
  if (d_pop1)
    cudaFree(d_pop1);
  if (d_pop2)
    cudaFree(d_pop2);
  if (d_pop3)
    cudaFree(d_pop3);

  if (d_count)
    cudaFree(d_count);
  if (d_count1)
    cudaFree(d_count1);
  if (d_count2)
    cudaFree(d_count2);
  if (d_count3)
    cudaFree(d_count3);

  if (d_fit)
    cudaFree(d_fit);
  if (d_fit1)
    cudaFree(d_fit1);
  if (d_fit2)
    cudaFree(d_fit2);
  if (d_fit3)
    cudaFree(d_fit3);
}
