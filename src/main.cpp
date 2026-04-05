#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <map>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include "pha_fim_gpu.hpp"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#include <sys/resource.h>
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

int main(int argc , char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    if (argc < 4) return 1;
    float min_sup_rate = stof(argv[1]);
    string input_file = argv[2];
    string output_file = argv[3];

    auto start_time = chrono::high_resolution_clock::now();

    // -----------------------
    // Pass 1 (FIRST READ)
    // -----------------------
    ifstream infile(input_file);
    if(!infile.is_open()) return 1;

    string pass1_tmp = output_file + ".pass1.tmp";
    ofstream tmpout(pass1_tmp);
    if (!tmpout.is_open()) return 1;

    map<int, int> temp_counts; // 1-item support
    string line;

    long long N = 0; // number of transactions
    while(getline(infile, line)){
        if(line.empty()) continue;
        stringstream ss(line);

        int item_id;
        vector<int> transaction;
        while(ss >> item_id) {
            transaction.push_back(item_id);
            if(item_id > max_id_found) max_id_found = item_id;
        }
        if (transaction.empty()) continue;

        for (int id : transaction) {
            temp_counts[id]++;
        }

        tmpout << (int)transaction.size();
        for (int id : transaction) tmpout << " " << id;
        tmpout << "\n";

        N++;
    }
    infile.close();
    tmpout.close();

    int min_sup = (int)ceil(min_sup_rate * (double)N);

    vector<char> is_frequent(max_id_found + 1, 0);

    vector<pair<int,int>> freq_items;
    freq_items.reserve(temp_counts.size());

    for (auto const& kv : temp_counts) {
        int id = kv.first;
        int sup = kv.second;
        if (sup >= min_sup && id >= 0 && id <= max_id_found) {
            is_frequent[id] = 1;
            freq_items.push_back({id, sup});
        }
    }

    sort(freq_items.begin(), freq_items.end(),
        [](const pair<int,int>& a, const pair<int,int>& b){
            if (a.second == b.second) return a.first < b.first;
            return a.second > b.second;
        });

    vector<int> old2new(max_id_found + 1, -1);
    for (int new_id = 0; new_id < (int)freq_items.size(); ++new_id) {
        old2new[freq_items[new_id].first] = new_id;
    }
    int m = (int)freq_items.size();

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
        // For now, we fall back to "no sort" (still correct, just less optimal for load balancing).
        sorted_tmp = pass1_tmp;
    #endif

    // -----------------------
    // Pass 2 (SECOND READ)
    // -----------------------
    vector<int> items_flat;
    vector<int> start, tx_len;
    items_flat.reserve(1 << 20);
    start.reserve(1 << 20);
    tx_len.reserve(1 << 20);
    int cursor = 0;

    ifstream sorted_in(sorted_tmp);
    if (!sorted_in.is_open()) return 1;

    ofstream outfile(output_file);
    if (!outfile.is_open()) return 1;

    long long kept_tx = 0;

    while (getline(sorted_in, line)) {
        if (line.empty()) continue;
        stringstream ss(line);

        int len_dummy;
        ss >> len_dummy; // first field = length

        vector<int> filtered;
        filtered.reserve(max(0, len_dummy));

        int id;
        while (ss >> id) {
            if (id >= 0 && id <= max_id_found && is_frequent[id]) {
                int nid = old2new[id];
                filtered.push_back(nid);
            }
        }

        if (!filtered.empty()) {
            // pack
            start.push_back(cursor);
            tx_len.push_back((int)filtered.size());
            items_flat.insert(items_flat.end(), filtered.begin(), filtered.end());
            cursor += (int)filtered.size();

            // debug
            for (int i = 0; i < (int)filtered.size(); ++i) {
                outfile << filtered[i] << (i + 1 == (int)filtered.size() ? "" : " ");
            }
            outfile << "\n";

            kept_tx++;
        }
    }

    sorted_in.close();
    outfile.close();

    int N_effective = (int)start.size();
    int P = 1024;
    int iters = 50;
    uint64_t seed = 42;

    run_gpu_fim(items_flat, start, tx_len,
                N_effective, m, min_sup,
                P, iters);

    auto end_time = chrono::high_resolution_clock::now();
    long final_memory = get_memory_usage();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "===== Performance Report =====\n";
    cout << "Time Elapsed: " << duration.count() << " ms\n";
    cout << "Memory Usage (Peak): " << final_memory << " KB\n";
    cout << "N (transactions): " << N << "\n";
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