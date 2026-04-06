#include "pha_fim_gpu.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <psapi.h>
#include <windows.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

using namespace std;

int max_id_found = 0;

long get_memory_usage() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return pmc.PeakWorkingSetSize / 1024;
  }
  return 0;
#else
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    return usage.ru_maxrss;
  }
  return 0;
#endif
}

SMItemset build_smitemset(const string &name,
                          const vector<pair<int, int>> &sorted_items,
                          int max_id_found) {
  SMItemset sm;
  sm.name = name;
  sm.order_info = sorted_items;
  sm.old2new.assign(max_id_found + 1, -1);
  sm.new2old.assign(sorted_items.size(), -1);

  for (int new_id = 0; new_id < (int)sorted_items.size(); ++new_id) {
    int old_id = sorted_items[new_id].first;
    sm.old2new[old_id] = new_id;
    sm.new2old[new_id] = old_id;
  }
  return sm;
}

int main(int argc, char *argv[]) {
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  if (argc < 4)
    return 1;
  float min_sup_rate = stof(argv[1]);
  string input_file = argv[2];
  string output_file = argv[3];

  auto start_time = chrono::high_resolution_clock::now();

  // -----------------------
  // Pass 1 (FIRST READ)
  // -----------------------
  ifstream infile(input_file);
  if (!infile.is_open())
    return 1;

  string pass1_tmp = output_file + ".pass1.tmp";
  ofstream tmpout(pass1_tmp);
  if (!tmpout.is_open())
    return 1;

  map<int, int> temp_counts; // 1-item support
  string line;

  long long trans_cnt = 0; // number of transactions
  while (getline(infile, line)) {
    if (line.empty())
      continue;
    stringstream ss(line);

    int item_id;
    vector<int> transaction;
    while (ss >> item_id) {
      transaction.push_back(item_id);
      if (item_id > max_id_found)
        max_id_found = item_id;
    }
    if (transaction.empty())
      continue;

    for (int id : transaction) {
      temp_counts[id]++;
    }

    tmpout << (int)transaction.size();
    for (int id : transaction)
      tmpout << " " << id;
    tmpout << "\n";

    trans_cnt++;
  }
  infile.close();
  tmpout.close();

  int min_sup = (int)ceil(min_sup_rate * (double)trans_cnt);

  vector<char> is_frequent(max_id_found + 1, 0);

  vector<pair<int, int>> freq_items;
  freq_items.reserve(temp_counts.size());

  for (auto const &kv : temp_counts) {
    int id = kv.first;
    int sup = kv.second;
    if (sup >= min_sup && id >= 0 && id <= max_id_found) {
      is_frequent[id] = 1;
      freq_items.push_back({id, sup});
    }
  }

  SMItemset sm1, sm2, sm3;
  vector<pair<int, int>> items1 = freq_items, items2 = freq_items,
                         items3 = freq_items;

  // sm1: support descending, item id ascending
  sort(items1.begin(), items1.end(),
       [](const pair<int, int> &a, const pair<int, int> &b) {
         if (a.second == b.second)
           return a.first < b.first;
         return a.second > b.second;
       });

  // sm2: support descending, item id descending
  sort(items2.begin(), items2.end(),
       [](const pair<int, int> &a, const pair<int, int> &b) {
         if (a.second == b.second)
           return a.first > b.first;
         return a.second > b.second;
       });

  // sm3: support ascending, item id ascending
  sort(items3.begin(), items3.end(),
       [](const pair<int, int> &a, const pair<int, int> &b) {
         if (a.second == b.second)
           return a.first < b.first;
         return a.second < b.second;
       });

  sm1 = build_smitemset("support_desc", items1, max_id_found);
  sm2 = build_smitemset("support_desc_rev_id", items2, max_id_found);
  sm3 = build_smitemset("support_asc", items3, max_id_found);

  int m = (int)sm1.order_info.size();

  string sorted_tmp = output_file + ".sorted_by_len.tmp";

#ifndef _WIN32
  {
    string cmd = "sort -n -k1,1 " + pass1_tmp + " -o " + sorted_tmp;
    int rc = system(cmd.c_str());
    if (rc != 0) {
      cerr << "[error] external sort failed. cmd=" << cmd << "\n";
      return 1;
    }
  }
#else
  // On Windows without GNU sort, you need another external sort method.
  // For now, we fall back to "no sort" (still correct, just less optimal for
  // load balancing).
  sorted_tmp = pass1_tmp;
#endif

  // -----------------------
  // Pass 2 (SECOND READ)
  // -----------------------
  PackedDataset ds1, ds2, ds3;

  ds1.items_flat.reserve(1 << 20);
  ds2.items_flat.reserve(1 << 20);
  ds3.items_flat.reserve(1 << 20);

  ds1.start.reserve(trans_cnt);
  ds2.start.reserve(trans_cnt);
  ds3.start.reserve(trans_cnt);

  ds1.tx_len.reserve(trans_cnt);
  ds2.tx_len.reserve(trans_cnt);
  ds3.tx_len.reserve(trans_cnt);

  int ptr1 = 0, ptr2 = 0, ptr3 = 0;

  ifstream sorted_in(sorted_tmp);
  if (!sorted_in.is_open())
    return 1;

  ofstream outfile(output_file);
  if (!outfile.is_open())
    return 1;

  long long kept_tx = 0;

  while (getline(sorted_in, line)) {
    if (line.empty())
      continue;
    stringstream ss(line);

    int len_dummy;
    ss >> len_dummy;

    vector<int> filtered1, filtered2, filtered3;
    filtered1.reserve(max(0, len_dummy));
    filtered2.reserve(max(0, len_dummy));
    filtered3.reserve(max(0, len_dummy));

    int id;
    while (ss >> id) {
      if (id >= 0 && id <= max_id_found && is_frequent[id]) {
        filtered1.push_back(sm1.old2new[id]);
        filtered2.push_back(sm2.old2new[id]);
        filtered3.push_back(sm3.old2new[id]);
      }
    }

    if (!filtered1.empty()) {
      ds1.start.push_back(ptr1);
      ds1.tx_len.push_back((int)filtered1.size());
      ds1.items_flat.insert(ds1.items_flat.end(), filtered1.begin(),
                            filtered1.end());
      ptr1 += (int)filtered1.size();

      ds2.start.push_back(ptr2);
      ds2.tx_len.push_back((int)filtered2.size());
      ds2.items_flat.insert(ds2.items_flat.end(), filtered2.begin(),
                            filtered2.end());
      ptr2 += (int)filtered2.size();

      ds3.start.push_back(ptr3);
      ds3.tx_len.push_back((int)filtered3.size());
      ds3.items_flat.insert(ds3.items_flat.end(), filtered3.begin(),
                            filtered3.end());
      ptr3 += (int)filtered3.size();

      kept_tx++;
    }
  }

  sorted_in.close();
  outfile.close();

  int N_effective = (int)ds1.start.size();
  int P = 1024;
  int iters = 50;

  run_gpu_fim(ds1, ds2, ds3, N_effective, m, min_sup, P, iters);

  auto end_time = chrono::high_resolution_clock::now();
  long final_memory = get_memory_usage();
  auto duration =
      chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

  cout << "===== Performance Report =====\n";
  cout << "Time Elapsed: " << duration.count() << " ms\n";
  cout << "Memory Usage (Peak): " << final_memory << " KB\n";
  cout << "N (transactions): " << trans_cnt << "\n";
  cout << "min_sup_count: " << min_sup << "\n";
  // cout << "frequent_1_items: " << headers.size() << "\n";
  cout << "kept_transactions: " << kept_tx << "\n";
  cout << "pass1_tmp: " << pass1_tmp << "\n";
#ifndef _WIN32
  cout << "sorted_tmp: " << sorted_tmp << "\n";
#else
  cout << "sorted_tmp: (windows fallback, not sorted)\n";
#endif
  cout << "output_file(filtered): " << output_file << "\n";
  cout << "==============================\n";

  return 0;
}
