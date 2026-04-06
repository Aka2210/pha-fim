#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace std;

struct SMItemset {
  string name;
  vector<pair<int, int>> order_info;
  vector<int> old2new;
  vector<int> new2old;
};

struct PackedDataset {
  vector<int> items_flat;
  vector<int> start;
  vector<int> tx_len;
};

struct DevicePackedDataset {
  int *d_items = nullptr;
  int *d_start = nullptr;
  int *d_len = nullptr;
  int items_count = 0;
  int tx_count = 0;
};

SMItemset build_smitemset(const string &name,
                          const vector<pair<int, int>> &sorted_items,
                          int max_id_found);

DevicePackedDataset upload_packed_dataset_to_gpu(const PackedDataset &ds);
void free_device_packed_dataset(DevicePackedDataset &dds);

void run_gpu_fim(const PackedDataset &ds1, const PackedDataset &ds2,
                 const PackedDataset &ds3, int N, int m, int min_sup_count,
                 int P, int iters, vector<uint8_t> &h_pop1, vector<int> &h_fit1,
                 vector<uint8_t> &h_pop2, vector<int> &h_fit2,
                 vector<uint8_t> &h_pop3, vector<int> &h_fit3);
