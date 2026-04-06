# PHA-FIM (Degraded from PHA-HUIM) — GPU Parallel Heuristic Frequent Itemset Mining

## 📌 Overview
This project implements a **GPU-parallel heuristic baseline** inspired by the IEEE TKDE 2024 paper:

> **"GPU-Based Efficient Parallel Heuristic Algorithm for High-Utility Itemset Mining in Large Transaction Datasets"**
> (Wei Fang et al., IEEE TKDE, 2024)

The original paper targets **High-Utility Itemset Mining (HUIM)**.
In this repo, we **degrade the method to classic Frequent Itemset Mining (FIM)** by using:

- **Fitness = Support (frequency)**
- **minUtility → minSup**

Conceptually, this is equivalent to setting **all item utilities to 1** and treating:
- `utility(itemset) = support(itemset)` (in transaction-count sense)

> ⚠️ Note: The PHA framework is **heuristic** (population-based).
> It does **not guarantee** enumerating *all* frequent itemsets like FP-Growth/Eclat.
> We evaluate **runtime** and **mining quality** (how many frequent patterns are found) against exact baselines.

---

## 🚀 Quick Start

### 1. Environment Setup
We provide a setup script that automates:

- System dependency checks (`g++`, **CUDA / nvcc**)
- Python virtual environment creation
- Library installation (`psutil` for resource monitoring)
- C++ / CUDA core compilation

```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

### 2. Data Preparation
Place raw transaction datasets in `data_raw/`.

**Supported formats:** `.data`, `.txt`
**Format:** items separated by commas (e.g., `f,n,t,l,won`)

The preprocessing automatically applies **Set Semantics**:
- Duplicate items in the same transaction are merged/removed before mining.

---

### 3. Running Automated Experiments
Activate the environment:

```bash
source .venv/bin/activate
```

#### Full Baseline Reproduction (example)
```bash
python3 experiment.py \
  --datasets "mushroom,connect4,car,kr-vs-kp" \
  --tx-ratios "10,50,100" \
  --minsup-ratios "1,2,5" \
  --override-default-minsup "mushroom=5,connect4=10" \
  --parallel 4 \
  --resume
```

To include **PHA-FIM** in the run, ensure your runner supports it (example):
```bash
python3 experiment.py \
  --datasets "mushroom,connect4" \
  --tx-ratios "10,50,100" \
  --minsup-ratios "1,2,5" \
  --baselines "FPGrowth_itemsets,Eclat,CICLAD,Hamm,PHA" \
  --parallel 4 \
  --resume
```

---

## 🧩 What “Degraded to FIM” Means
The original paper optimizes a HUIM heuristic that searches for **high-utility** patterns using:

- **Population-based search**
- **GPU-parallel fitness evaluation**
- **Ring-topology communication**
- **Sort-mapping compression + load balancing**

In HUIM, `fitness(X) = utility(X)` and threshold is `minUtility`.
In our FIM degradation:

- `fitness(X) = support(X)`
- threshold is `minSupCount = ceil(minsup_rate * N)` where `N` is #transactions

Everything else (search strategy, ring topology, GPU load balancing, compact storage) stays aligned with the paper’s framework.

---

## 🛠️ Implementation Logic (Paper Framework → FIM Version)

### 0. Data Layout: SM + 1D Compressed Dataset (GPU-friendly)
Following the paper design, transactions are stored as:

- `items_flat[]` (all transactions concatenated)
- `start[i]`, `len[i]` to locate transaction `i`

This avoids pointer-heavy `vector<vector<int>>` structures on GPU and improves memory coalescing.

---

### 1. CPU Preprocessing (Two-Pass Style)
The paper performs preprocessing on CPU before copying data to GPU.

In our FIM degradation, the CPU stage typically includes:

1. **Scan 1**: count item supports; optionally record transaction lengths
2. **Compute minSupCount**: `ceil(minsup_rate * N)`
3. **Prune 1-infrequent items**: remove items with support < minSupCount
4. **Sort-Mapping (SM)**: map remaining items to dense IDs `0..m-1`
   (and optionally build multiple SM orderings to mimic multi-start)
5. **Pack dataset** into `items_flat/start/len`

---

### 2. GPU Iteration: 3 Evolution Steps (Main Loop)
Each iteration runs **three GPU steps** (same as the paper’s framework):

#### Step A — MSUS Search (Multi-start + Unbalanced Allocation)
- Population is represented as **binary vectors / bitsets**
- Each iteration flips **one bit** per individual (local search)
- Multi-start runs multiple “views” (in HUIM: TWU/Sup/TU; in FIM: support-based variants)

#### Step B — Parallel Fitness Evaluation (Support Computation)
- Fitness is computed **in parallel on GPU**
- For each individual itemset `X`, scan transactions and count:
  - `support(X) = #{ T | X ⊆ T }`
- This is the most expensive step → accelerated with GPU parallelism

#### Step C — Ring Topology Communication
- Individuals communicate with neighbors in a ring structure
- Preserves strong candidates and maintains population diversity
- Produces the next-generation population

---

### 3. GPU Optimization: Load Balancing
Transactions vary in length → threads may diverge.
We adopt a **load balancing strategy** to distribute transaction workloads more evenly across threads/blocks, reducing divergence and improving throughput.

---

## 📋 Argument Reference (`experiment.py`)
| Argument | Description | Example |
|---|---|---|
| `--datasets` | Dataset names in `data_raw/` | `mushroom,car` |
| `--tx-ratios` | Percentage of transactions to use | `10,50,100` |
| `--tx-size` | Fixed number of transactions (overrides ratios) | `50000` |
| `--minsup-ratios` | Multipliers for base support | `0.5,1,2` |
| `--override-default-minsup` | Dataset-specific base minsup (%) | `mushroom=5` |
| `--parallel` / `--jobs` | Number of parallel processes | `4` |
| `--resume` | Skip already completed experiments | `--resume` |
| `--baselines` | Choose baselines to run | `FPGrowth_itemsets,Eclat,CICLAD,Hamm,PHA` |

---

## 📁 Project Structure
```text
src/                    # CPU preprocessing + host-side driver
cuda/                   # CUDA kernels: search / fitness / ring (and helpers)
include/                # Shared structs/types (dataset pack, bitset, params)
tools/pha_fim            # Compiled PHA-FIM executable (CUDA)
tools/hamm               # Compiled Hamm baseline (FP-Growth + Single-path opt)
tools/spmf.jar           # SPMF baselines (FP-Growth/Eclat)
tools/ciclad             # CICLAD baseline binary
experiment.py            # Python experiment runner
setup_env.sh             # Environment setup & build script
data_raw/                # Input datasets (.data or .txt)
results/                 # Outputs (metrics, plots, logs)
```

---

## 🧠 Conceptual Summary
This project provides a **baseline implementation** of the paper’s GPU-parallel heuristic framework,
but **degraded to FIM**:

- Keep: **population search + GPU fitness + ring topology + sort-mapping + load balancing**
- Replace: **utility → support** (HUIM → FIM)

This enables fair benchmarking against classic exact FIM baselines (FP-Growth/Eclat/Hamm/CICLAD)
under a unified `experiment.py` workflow.
