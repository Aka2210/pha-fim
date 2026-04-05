#pragma once
#include <vector>
#include <cstdint>

void run_gpu_fim(
    const std::vector<int>& items_flat,
    const std::vector<int>& start,
    const std::vector<int>& len,
    int N,
    int m, 
    int min_sup_count,  
    int P, 
    int iters 
);