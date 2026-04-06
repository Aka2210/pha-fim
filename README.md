# PHA-FIM Baseline — GPU Parallel Heuristic Frequent Itemset Mining

## Overview
This project implements a **GPU-parallel heuristic baseline for Frequent Itemset Mining (FIM)**, inspired by the framework of the IEEE TKDE 2024 paper:

> **GPU-Based Efficient Parallel Heuristic Algorithm for High-Utility Itemset Mining in Large Transaction Datasets**
> Wei Fang et al., IEEE TKDE, 2024

The original paper is designed for **High-Utility Itemset Mining (HUIM)**.
This project **degrades the framework to FIM** in order to study the efficiency of the framework after removing utility-specific components.

### Main degradation from HUIM to FIM
- `utility`-based fitness -> `support`-based fitness
- `minUtility` -> `minSupCount`
- HUIM-oriented item ordering -> FIM-compatible multi-order SM-itemsets

This implementation is intended as a **baseline for efficiency analysis**, not as a full reimplementation of the original HUIM paper.

---

## Current Project Status
This repository currently provides a **working FIM baseline** with the following components:

### Implemented
- CPU-side two-pass preprocessing
- 1-item support counting and infrequent-item pruning
- Three SM-itemsets (three remapped item orders)
- Packed transaction storage:
  - `items_flat`
  - `start`
  - `tx_len`
- GPU base population initialization
- Three-view search on GPU
- GPU support-based fitness evaluation
- Ring-topology-inspired population communication / merge
- Transaction-length sorting as a load-balancing approximation

### Not fully equivalent to the paper
This implementation is **inspired by** the paper’s framework, but is **not a full HUIM reproduction**.

Specifically:
- The mining task is **FIM**, not HUIM
- The three search views are **FIM-specific substitutes**, not the original TWU / Sup / TU HUIM design
- The ring topology step is implemented as a **practical CUDA approximation**
- The load balancing strategy follows the paper’s idea, but is **not a full paper-level greedy scheduling implementation**

---

## Design Goal
The goal of this project is:

> **To evaluate the efficiency of a degraded FIM version of the paper’s GPU population-based heuristic framework, and use it as a baseline.**

So the main focus is:
- runtime
- GPU execution behavior
- search efficiency after degradation to FIM

rather than exact reproduction of all theoretical details from the original HUIM method.

---

## Pipeline

### 1. CPU Preprocessing
The CPU stage performs two-pass preprocessing before entering the GPU phase.

#### Pass 1
- Read the dataset
- Count 1-item support
- Compute `minSupCount`
- Remove infrequent 1-items

#### SM-itemset generation
Three FIM-compatible SM-itemsets are built:
- `sm1`: support descending
- `sm2`: support descending with reversed tie order
- `sm3`: support ascending

These three item orders are used to simulate multi-view search behavior in the degraded FIM setting.

#### Load-balancing-oriented ordering
Before building GPU datasets, transactions are sorted by **transaction length**.

This is used to approximate the paper’s load balancing idea:
- nearby GPU threads process transactions of similar lengths
- workload imbalance and divergence are reduced during fitness evaluation

#### Pass 2
Build three packed datasets:
- `ds1`
- `ds2`
- `ds3`

Each dataset is remapped using one SM-itemset.

Each packed dataset contains:
- `items_flat`
- `start`
- `tx_len`

---

### 2. GPU Main Loop
The GPU stage executes an iterative heuristic mining process.

#### Step A — Base population initialization
A base population `d_pop` is initialized on GPU.

- Population size = `P`
- Each individual is a binary vector of length `m`
- `m` = number of frequent 1-items after pruning

#### Step B — Three-view branching
At each iteration, the base population is copied into:
- `d_pop1`
- `d_pop2`
- `d_pop3`

These three branches correspond to three search views.

#### Step C — Three-view search
A CUDA search kernel performs local mutation-like updates.

Current design:
- one thread = one individual
- each individual flips one bit
- population is divided into three search-range groups
- each branch uses a different view-aware search policy

This is the degraded FIM counterpart of the paper’s multi-start unbalanced search strategy.

#### Step D — Support-based fitness evaluation
For each individual:
- scan transactions
- count support
- fitness = support count

This replaces the paper’s utility-based fitness with FIM-style support evaluation.

#### Step E — Ring-topology-inspired merge
The three search branches are merged back into the base population using a ring-neighborhood comparison mechanism.

This step keeps the implementation closer to the paper’s communication idea while remaining practical for the current FIM baseline.

---

## Why This Is a Baseline
This project should be interpreted as:

- a **framework-level degradation** from HUIM to FIM
- a **GPU heuristic baseline**
- a platform for evaluating:
  - runtime
  - scalability
  - effect of three-view search
  - effect of GPU-side support evaluation

It should **not** be interpreted as:
- a complete HUIM implementation
- an exact reproduction of all paper-level details
- an exact frequent-itemset enumerator like FP-Growth or Eclat

Because this is still a **population-based heuristic method**, it does **not guarantee complete enumeration of all frequent itemsets**.

---

## Input Format
Current preprocessing expects transaction data as whitespace-separated integer item IDs, for example:

```text
1 3 5 8
2 4
1 2 3
```

---

## Build / Run
Adjust according to your local build setup. The current implementation consists of:

- `main.cpp`
- `pha_fim_gpu.hpp`
- `pha_fim_gpu.cu`

A typical build uses:
- `g++` for host code
- `nvcc` for CUDA compilation

---

## Core Data Structures

### CPU-side
- `SMItemset`
- `PackedDataset`

### GPU-side
- `DevicePackedDataset`
- `d_pop`, `d_pop1`, `d_pop2`, `d_pop3`
- `d_count`, `d_count1`, `d_count2`, `d_count3`
- `d_fit`, `d_fit1`, `d_fit2`, `d_fit3`

---

## Experimental Positioning
This implementation is best used as:

- a **degraded FIM baseline**
- a **runtime comparison target**
- a **GPU heuristic reference point**

for later comparison against:
- exact FIM algorithms
- other GPU/parallel baselines
- improved versions of the framework

---

## Summary
This repository currently implements:

- a **GPU-parallel heuristic FIM baseline**
- structurally inspired by the PHA-HUIM paper
- but explicitly **degraded from HUIM to FIM**

It keeps the framework ideas:
- multi-view SM-itemsets
- GPU population search
- support-based fitness on GPU
- ring-style communication
- load-balancing-oriented transaction ordering

while replacing HUIM-specific utility logic with FIM support logic.

This makes it suitable for studying:

> **how the paper-inspired GPU heuristic framework behaves after degradation to FIM, especially in terms of efficiency.**
