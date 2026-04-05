# -*- coding: utf-8 -*-
# Baseline experiment runner for frequent itemset mining (SPMF: FP-Growth/Eclat, CICLAD).
#
# Highlights:
#   - Deterministic preprocessing into SPMF (.spmf) and CICLAD (.dat) formats
#   - Parallel execution across datasets and tx/minsup sweep points
#   - Optional FIFO streaming to avoid writing large pattern files to disk
#   - Resume/caching via metrics_*.json and per-point records
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIM Baseline Runner (FP-Growth, Eclat, CICLAD KDD'20 implementation)

This script:
  1) Preprocesses raw datasets into a transaction file where each line is a transaction:
       <int> <int> <int> ...
     (space-separated positive integer item IDs, one transaction per line)
     It writes:
       results/<ds>_transactions.spmf
       results/<ds>_transactions.dat      (same content; CICLAD input)
       results/<ds>_item2id.json          (token -> int mapping)
       results/<ds>_meta.json

  2) Runs baselines:
     - FP-Growth and Eclat via SPMF (spmf.jar)
     - CICLAD via ciclad_nogen binary (KDD'20 folder ciclad_nogen)

CICLAD binary interface assumed:
  ciclad_nogen <input.dat> <nbr_items> <window_size> <minsup1> <minsup2> ...
where minsup values are *support counts* (not percentages).

CICLAD writes:
  - frequent closed itemsets to STDOUT (we redirect to a file)
  - logs/timing to STDERR (we redirect to a log file)
We parse the log to get pattern counts per minsup count.

Environment variables (optional):
  PROJECT_DIR, DATA_DIR, RESULTS_DIR, SPMF_JAR, CICLAD_BIN, RANDOM_SEED

Raw datasets expected in DATA_DIR (default: ./data_raw):
  mushroom:    mushroom.csv OR agaricus-lepiota.data
  connect4:    connect-4.data
  tic-tac-toe: tic-tac-toe.data
  car:         car.data OR car.data.csv OR car-evaluation.data
  kr-vs-kp:    kr-vs-kp.data
"""

import os, json, random, subprocess, time, pathlib, argparse, shlex
import math, shutil, re

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt

import tempfile
import threading
import uuid


# ----------------------
# Paths & defaults (can be overridden by env vars)
# ----------------------
PROJECT_DIR = os.environ.get("PROJECT_DIR", os.getcwd())
DATA_DIR    = os.environ.get("DATA_DIR",    os.path.join(PROJECT_DIR, "data_raw"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(PROJECT_DIR, "results"))
SPMF_JAR    = os.environ.get("SPMF_JAR",    os.path.join(PROJECT_DIR, "tools", "spmf.jar"))
CICLAD_BIN  = os.environ.get("CICLAD_BIN",  os.path.join(PROJECT_DIR, "tools", "ciclad"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
JAVA_CMD    = os.environ.get("JAVA_CMD") or (shutil.which("java") or "java")

PATTERN_TMPDIR = os.environ.get("PATTERN_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"

def _tmp_pattern_path(prefix, suffix=".txt"):
    """Create a unique temp path for large pattern outputs (SPMF)."""
    try:
        os.makedirs(PATTERN_TMPDIR, exist_ok=True)
    except Exception:
        pass
    # Use mkstemp to avoid name collisions under parallel runs
    fd, p = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=PATTERN_TMPDIR)
    os.close(fd)
    return p
# Where to place FIFO files for streaming SPMF outputs.
# Default preference: PATTERN_FIFO_DIR env > TMPDIR env > ./results/_fifo_patterns
PATTERN_FIFO_DIR = os.environ.get("PATTERN_FIFO_DIR") or os.environ.get("TMPDIR") or os.path.join(os.getcwd(), "results", "_fifo_patterns")

def _mkfifo_pattern_path(prefix="spmf_", suffix=".fifo"):
    """Create a unique FIFO path (named pipe). The FIFO itself does NOT store data on disk."""
    os.makedirs(PATTERN_FIFO_DIR, exist_ok=True)
    name = f"{prefix}{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex}{suffix}"
    return os.path.join(PATTERN_FIFO_DIR, name)

def _parse_spmf_line_for_len(line: str) -> int:
    """Return item count in a SPMF output line like: '1 2 3 #SUP: 10'."""
    line = line.strip()
    if not line:
        return 0
    left = line.split("#SUP:")[0].strip()
    if not left:
        return 0
    return len(left.split())


def _parse_spmf_line_for_sup(line: str):
    """Parse support from a SPMF output line like: '1 2 3 #SUP: 10'. Return int or None."""
    line = line.strip()
    if not line:
        return None
    parts = line.split("#SUP:")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1].strip())
    except Exception:
        return None



DATASETS_ALL = ["mushroom","connect4","kr-vs-kp","tic-tac-toe","car"]

# Default x-axis points
DEFAULT_TX_RATIOS = [10,20,30,50,70,100]
DEFAULT_MINSUP_SWEEP = [0.5,1,2,5,10]

# Default fixed minsup (%) used in tx-ratio sweep
DEFAULT_MINSUP = {
    "mushroom":   1.0,
    "connect4":   1.0,
    "kr-vs-kp":   1.0,
    "tic-tac-toe":2.0,
    "car":        5.0,
}

# Algorithms we plot
ALGORITHMS = ["FPGrowth_itemsets", "Eclat", "CICLAD", "Hamm"]


# ----------------------
# Dataset loading & tokenization
# ----------------------
def one_hot_row(row, drop_cols=()):
    """
    Convert a row (pandas Series) into token items: "col=value".
    Notes:
      - Keeps '?' as a legitimate categorical value (do NOT drop it).
      - Drops empty/NaN.
    """
    items = []
    for col, val in row.items():
        if col in drop_cols:
            continue
        v = str(val).strip()
        if v in {"", "nan", "None"}:
            continue
        items.append(f"{col}={v}")
    return items


def _read_any_existing(paths, **pd_kwargs):
    for p in paths:
        if p and os.path.exists(p):
            return p, pd.read_csv(p, **pd_kwargs)
    raise FileNotFoundError(f"None of these files exist: {paths}")


def load_mushroom():
    # Prefer user-provided CSV with headers, else UCI .data (no header)
    p_csv = os.path.join(DATA_DIR, "mushroom.csv")
    p_dat = os.path.join(DATA_DIR, "agaricus-lepiota.data")

    if os.path.exists(p_csv):
        df = pd.read_csv(p_csv)
        # include label by default: drop = []
        drop = []
    else:
        df = pd.read_csv(p_dat, header=None)
        df.columns = list(range(df.shape[1]))
        drop = []  # include class label too (col 0)
    return [one_hot_row(r, drop_cols=drop) for _, r in df.iterrows()]


def load_connect4():
    path = os.path.join(DATA_DIR, "connect-4.data")
    df = pd.read_csv(path, header=None)
    df.columns = list(range(df.shape[1]))
    drop = []  # include label (last col)
    return [one_hot_row(r, drop_cols=drop) for _, r in df.iterrows()]


def load_tictactoe():
    path = os.path.join(DATA_DIR, "tic-tac-toe.data")
    df = pd.read_csv(path, header=None)
    df.columns = list(range(df.shape[1]))
    drop = []  # include label (last col)
    return [one_hot_row(r, drop_cols=drop) for _, r in df.iterrows()]


def load_car():
    candidates = [
        os.path.join(DATA_DIR, "car.data"),
        os.path.join(DATA_DIR, "car.data.csv"),
        os.path.join(DATA_DIR, "car-evaluation.data"),
        os.path.join(DATA_DIR, "car_evaluation.data"),
    ]
    p = None
    for c in candidates:
        if os.path.exists(c):
            p = c
            break
    if p is None:
        raise FileNotFoundError(f"Cannot find car dataset in {DATA_DIR}. Tried: {candidates}")

    df = pd.read_csv(p, header=None)
    df.columns = list(range(df.shape[1]))
    drop = []  # include label (last col)
    return [one_hot_row(r, drop_cols=drop) for _, r in df.iterrows()]


def load_kr_vs_kp():
    path = os.path.join(DATA_DIR, "kr-vs-kp.data")
    df = pd.read_csv(path, header=None)
    df.columns = list(range(df.shape[1]))
    drop = []  # include label if present
    return [one_hot_row(r, drop_cols=drop) for _, r in df.iterrows()]


LOADERS = {
    "mushroom": load_mushroom,
    "connect4": load_connect4,
    "tic-tac-toe": load_tictactoe,
    "car": load_car,
    "kr-vs-kp": load_kr_vs_kp,
}


def build_item2id_scan_order(transactions):
    """
    Build a token-to-integer mapping using a deterministic scan order.

    Input:
      - transactions: list[list[str]] where each item is already a token string.

    Scan order:
      - Traverse transactions in order (top to bottom).
      - Within each transaction, traverse tokens in order (left to right).

    Numbering rule:
      - Assign a new integer ID the first time a token is observed.
      - The mapping is stable for a fixed preprocessing order and can be inverted
        via the returned id2item list.

    Important:
      - Tokens should be column-qualified (e.g., "col=value") for categorical datasets
        so that identical symbols in different columns (e.g., 'f' in chess/kr-vs-kp)
        are treated as distinct items.
    """
    item2id = {}
    next_id = 1
    for tx in transactions:
        for tok in tx:
            if tok not in item2id:
                item2id[tok] = next_id
                next_id += 1
    return item2id


def write_transactions_int(transactions, out_path, item2id):
    """
    Write each transaction as sorted integer IDs (ascending), one per line.
    (Sort for canonical form; numbering itself is scan-order.)
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for tx in transactions:
            ids = sorted({item2id[t] for t in tx})
            f.write(" ".join(map(str, ids)) + "\n")


def scan_max_id(path):
    max_id = 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            for tok in line.split():
                try:
                    v = int(tok)
                    if v > max_id:
                        max_id = v
                except ValueError:
                    pass
    return n, max_id


# ----------------------
# File utilities
# ----------------------
def safe_unlink(path):
    """Best-effort delete a file (ignore errors)."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass



# ----------------------
# Tool runners
# ----------------------

def run_spmf(algorithm, input_path, output_path, minsup_percent, keep_pattern_files=False, minsup_count_filter=None):
    """
    Run SPMF algorithm with minsup as percent (e.g. 1.0 -> "1.0%").
    Return runtime + output stats.

    Disk-usage optimized behavior:
      - keep_pattern_files=True: write patterns to output_path (regular file) and KEEP it.
      - keep_pattern_files=False (default): stream patterns via a FIFO (named pipe), count patterns and max itemset length
        on-the-fly, and do NOT store the pattern output on disk.

    Fixed minsup-count support (for tx-ratio sweep):
      - minsup_count_filter: Optional[int]
          If provided, we post-filter patterns by absolute support count (#SUP) >= minsup_count_filter.
          This is used for tx-ratio sweep mode "count" where minsup is a fixed *count* computed from N_full.

    Notes:
      - FIFO requires a filesystem that supports named pipes. If your filesystem does not, set PATTERN_FIFO_DIR to a writable dir.
    """
    if not os.path.isfile(SPMF_JAR):
        raise FileNotFoundError(f"spmf.jar not found at: {SPMF_JAR}")

    minsup_arg = f"{float(minsup_percent)}%"
    java_prefix = shlex.split(JAVA_CMD)

    def _accept_line(line: str) -> bool:
        if minsup_count_filter is None:
            return True
        sup = _parse_spmf_line_for_sup(line)
        if sup is None:
            return False
        return sup >= int(minsup_count_filter)

    # If user wants to keep patterns, use normal file output.
    if keep_pattern_files:
        cmd = java_prefix + ["-Djava.awt.headless=true", "-jar", SPMF_JAR, "run", algorithm, input_path, output_path, minsup_arg]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"SPMF {algorithm} failed (JAVA_CMD={JAVA_CMD}):\n{proc.stderr}")

        count, max_len = 0, 0
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if not _accept_line(line):
                        continue
                    count += 1
                    max_len = max(max_len, _parse_spmf_line_for_len(line))
        return {"runtime_sec": elapsed, "pattern_count": count, "max_itemset_len": max_len, "cmd": " ".join(cmd)}

    # Default: FIFO streaming (no large disk output)
    fifo_path = _mkfifo_pattern_path(prefix=f"spmf_{algorithm}_")
    try:
        if os.path.exists(fifo_path):
            safe_unlink(fifo_path)
        os.mkfifo(fifo_path, 0o600)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create FIFO at {fifo_path}. "
            f"Try setting PATTERN_FIFO_DIR to a writable filesystem. Error: {e}"
        )

    stats = {"count": 0, "max_len": 0}
    read_err = {"err": None}

    def _reader():
        try:
            with open(fifo_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if not _accept_line(line):
                        continue
                    stats["count"] += 1
                    stats["max_len"] = max(stats["max_len"], _parse_spmf_line_for_len(line))
        except Exception as e:
            read_err["err"] = e

    th = threading.Thread(target=_reader, daemon=True)
    th.start()

    cmd = java_prefix + ["-Djava.awt.headless=true", "-jar", SPMF_JAR, "run", algorithm, input_path, fifo_path, minsup_arg]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - t0

    # Wait for reader to finish draining FIFO after writer closes
    th.join(timeout=120.0)

    safe_unlink(fifo_path)

    if read_err["err"] is not None:
        raise RuntimeError(f"SPMF FIFO reader failed: {read_err['err']}")

    if proc.returncode != 0:
        raise RuntimeError(f"SPMF {algorithm} failed (JAVA_CMD={JAVA_CMD}):\n{proc.stderr}")

    return {"runtime_sec": elapsed, "pattern_count": stats["count"], "max_itemset_len": stats["max_len"], "cmd": " ".join(cmd)}

def run_hamm(input_path, output_path, minsup_percent, keep_pattern_files=False):
    hamm_bin = os.path.join(PROJECT_DIR, "tools", "hamm")
    
    minsup_rate = float(minsup_percent) / 100.0
    
    cmd = [hamm_bin, str(minsup_rate), input_path, output_path]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"Hamm failed: {proc.stderr}")

        runtime_sec = elapsed
        for line in proc.stdout.splitlines():
            if "Time Elapsed:" in line:
                runtime_sec = float(line.split(":")[1].strip().split()[0]) / 1000.0

        count = 0
        max_len = 0
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        count += 1
                        max_len = max(max_len, len(line.split("#SUP:")[0].split()))
            
            if not keep_pattern_files:
                safe_unlink(output_path)

        return {
            "runtime_sec": runtime_sec, 
            "pattern_count": count, 
            "max_itemset_len": max_len, 
            "cmd": " ".join(cmd)
        }
    except Exception as e:
        raise RuntimeError(f"Hamm execution error: {e}")

def parse_ciclad_log(log_path):
    """
    Parse CICLAD stderr log.

    Expected patterns (from your log.txt):
      minsup_counts: 82,102,152,...
      dumped frequent closed itemsets: 51640
      Minsup: 82
      processed transactions in 96543.2660 ms
    """
    minsups = []
    dumped = []
    times_ms = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()

            m = re.match(r"^minsup_counts:\s*(.+)$", line)
            if m:
                minsups = [int(x) for x in m.group(1).split(",") if x.strip()]
                continue

            m = re.search(r"processed transactions in\s*([0-9.]+)\s*ms", line)
            if m:
                try:
                    times_ms.append(float(m.group(1)))
                except Exception:
                    pass
                continue

            m = re.search(r"dumped frequent closed itemsets:\s*(\d+)", line)
            if m:
                dumped.append(int(m.group(1)))
                continue

    dumped_by = {}
    if minsups:
        for i, ms in enumerate(minsups):
            if i < len(dumped):
                dumped_by[ms] = dumped[i]
    else:
        # fallback: index by order
        for i, d in enumerate(dumped):
            dumped_by[i] = d

    return {
        "minsup_counts": minsups,
        "dumped_fci_by_minsup": dumped_by,
        "raw_times_ms": times_ms,
        "raw_dumped_list": dumped,
    }


def run_ciclad_multi(input_dat, fci_out_path, log_path, nbr_items, window_size, minsup_counts, keep_pattern_files=False):
    """
    Run CICLAD once with multiple minsup counts.
    - If keep_pattern_files=True, writes frequent closed itemsets to fci_out_path (STDOUT).
    - Always writes log to log_path (STDERR) and parses it to get counts.
    - Returns parsed log + wall-clock runtime.

    When keep_pattern_files=False (default), STDOUT is discarded (sent to /dev/null) to save disk space.
    """
    if (not os.path.isfile(CICLAD_BIN)) or (not os.access(CICLAD_BIN, os.X_OK)):
        raise FileNotFoundError(f"CICLAD binary not found/executable at: {CICLAD_BIN}")

    cmd = [CICLAD_BIN, str(input_dat), str(int(nbr_items)), str(int(window_size))] + [str(int(x)) for x in minsup_counts]

    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as f_log:
        if keep_pattern_files:
            with open(fci_out_path, "w", encoding="utf-8") as f_out:
                proc = subprocess.run(cmd, stdout=f_out, stderr=f_log, text=True)
        else:
            # Avoid writing huge FCI outputs to disk; we only need STDERR log for counts.
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=f_log, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        if not keep_pattern_files:
            safe_unlink(fci_out_path)
        raise RuntimeError(f"CICLAD failed (rc={proc.returncode}). CMD: {' '.join(cmd)}")

    parsed = parse_ciclad_log(log_path)
    parsed["runtime_sec_wall"] = elapsed
    parsed["cmd"] = " ".join(cmd)

    if not keep_pattern_files:
        safe_unlink(fci_out_path)

    return parsed

def subsample_lines(input_path, output_path, ratio_percent, seed=42):
    """
    Randomly subsample transactions (without replacement) to ratio_percent.
    Works for .spmf or .dat (same line format).
    Returns number of lines written.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    n = len(lines)
    k = max(1, int(round(n * (ratio_percent / 100.0))))
    rng = random.Random(seed + int(ratio_percent))
    idx = list(range(n))
    rng.shuffle(idx)
    sel = sorted(idx[:k])

    out_path = pathlib.Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in sel:
            f.write(lines[i] + "\n")

    return k


def plot_multi(x, ydict, title, xlabel, ylabel, out_png):
    plt.figure()
    for label, y in ydict.items():
        plt.plot(x, y, marker="o", label=label)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()


def load_metrics_if_any(ds_dir, ds_name):
    p = os.path.join(ds_dir, f"metrics_{ds_name}.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"dataset": ds_name, "by_txratio": [], "by_minsup": []}
    return {"dataset": ds_name, "by_txratio": [], "by_minsup": []}



def find_cached(point_list, **kv):
    """
    Flexible matcher for legacy + new records.

    Legacy records may not have:
      - tx_sweep_minsup_mode (default: "percent")
      - minsup_count_threshold

    For minsup_count_threshold matching:
      - if record lacks minsup_count_threshold, try record["minsup_count"] (legacy CICLAD)
      - if still missing and requested/record mode is "percent", ignore threshold mismatch (treat as match)
    """
    req_mode = kv.get("tx_sweep_minsup_mode", None)
    req_mode = (str(req_mode).lower() if req_mode is not None else None)

    for rec in point_list:
        ok = True
        rec_mode = str(rec.get("tx_sweep_minsup_mode", "percent")).lower()

        for k, v in kv.items():
            if k == "tx_sweep_minsup_mode":
                if str(rec_mode) != str(req_mode):
                    ok = False
                    break
                continue

            if k == "minsup_count_threshold":
                rec_v = rec.get("minsup_count_threshold", None)
                if rec_v is None and rec.get("minsup_count", None) is not None:
                    rec_v = rec.get("minsup_count")
                mode_eff = req_mode or rec_mode
                if rec_v is None and v is not None and mode_eff == "percent":
                    continue
                if str(rec_v) != str(v):
                    ok = False
                    break
                continue

            if str(rec.get(k)) != str(v):
                ok = False
                break

        if ok:
            return rec
    return None


def build_resume_cache(metrics):
    """
    Build fast lookup caches for --resume mode.

    Legacy tx key (v2):
      (alg, tx_ratio, minsup_percent)

    New tx key (v3):
      (alg, tx_ratio, tx_sweep_minsup_mode, minsup_percent, minsup_count_threshold_or_None)

    We store BOTH so old metrics JSON remains reusable.
    """
    cache_tx = {}

    for rec in metrics.get("by_txratio", []):
        try:
            alg = rec.get("algorithm")
            txr = float(rec.get("transaction_ratio_percent"))
            msp = float(rec.get("minsup_percent"))

            cache_tx[(alg, txr, msp)] = rec  # legacy

            mode = str(rec.get("tx_sweep_minsup_mode", "percent")).lower()

            mct = rec.get("minsup_count_threshold", None)
            if mct is None:
                if rec.get("minsup_count", None) is not None:
                    mct = rec.get("minsup_count")
                elif rec.get("n_transactions_sub", None) is not None:
                    try:
                        n_sub = int(rec.get("n_transactions_sub"))
                        mct = int(math.ceil(msp / 100.0 * n_sub))
                    except Exception:
                        mct = None
            mct = int(mct) if mct is not None else None

            cache_tx[(alg, txr, mode, msp, mct)] = rec
            if mct is None:
                cache_tx[(alg, txr, mode, msp, None)] = rec
        except Exception:
            pass

    cache_ms = {}
    for rec in metrics.get("by_minsup", []):
        try:
            alg = rec.get("algorithm")
            ms = float(rec.get("minsup_percent"))
            mc = rec.get("minsup_count", None)
            mc = int(mc) if mc is not None else None
            cache_ms[(alg, ms, mc)] = rec
        except Exception:
            pass

    return cache_tx, cache_ms


def worker_txratio_point(ds, r, spmf_path, nbr_items, n_tx_full, ms_default, tx_sweep_minsup_mode, baselines, ds_dir, resume, cache_tx, keep_pattern_files):
    """
    Run selected baselines for one (dataset, tx_ratio) point.

    tx_sweep_minsup_mode:
      - "percent": fixed minsup ratio (default). Threshold count scales with n_sub.
      - "count"  : fixed minsup COUNT computed from full dataset size (100% tx * minsup ratio).
                  For SPMF, we pass an effective percent and post-filter patterns by support >= fixed_count.

    Returns: (ds, r, recs[list-of-dicts])
    """
    r = float(r)
    mode = str(tx_sweep_minsup_mode).strip().lower()
    if mode not in {"percent", "count"}:
        mode = "percent"

    sub_path = os.path.join(ds_dir, f"sub_{int(r)}.spmf")
    if not os.path.exists(sub_path):
        n_sub = subsample_lines(spmf_path, sub_path, r, seed=RANDOM_SEED)
    else:
        n_sub, _ = scan_max_id(sub_path)

    n_sub = int(n_sub)
    n_tx_full = int(n_tx_full)

    if mode == "percent":
        minsup_count_threshold = int(math.ceil(ms_default / 100.0 * n_sub))
        effective_minsup_percent = float(ms_default)
        fixed_minsup_count = None
        base_n_tx_for_fixed = None
        spmf_filter_count = None
    else:
        fixed_minsup_count = int(math.ceil(ms_default / 100.0 * max(1, n_tx_full)))
        base_n_tx_for_fixed = n_tx_full
        minsup_count_threshold = int(fixed_minsup_count)

        raw_eff = 100.0 * float(fixed_minsup_count) / float(max(1, n_sub))
        effective_minsup_percent = float(raw_eff) if raw_eff <= 100.0 else 100.0
        spmf_filter_count = int(fixed_minsup_count)

    recs = []

    def _cache_lookup(alg_name: str):
        if not resume:
            return None
        k_new = (alg_name, float(r), mode, float(ms_default), int(minsup_count_threshold) if minsup_count_threshold is not None else None)
        if k_new in cache_tx:
            return cache_tx[k_new]
        k_relaxed = (alg_name, float(r), mode, float(ms_default), None)
        if k_relaxed in cache_tx:
            return cache_tx[k_relaxed]
        if mode == "percent":
            k_old = (alg_name, float(r), float(ms_default))
            if k_old in cache_tx:
                return cache_tx[k_old]
        return None

    for alg in [a for a in baselines if a in {"FPGrowth_itemsets", "Eclat", "Hamm"}]:
        cached = _cache_lookup(alg)
        if cached is not None:
            recs.append(cached)
            continue

        out_file = os.path.join(ds_dir, f"{alg}_tx{int(r)}.txt")
        if alg == "Hamm":
            m = run_hamm(sub_path, out_file, effective_minsup_percent, keep_pattern_files)
        else:
            m = run_spmf(alg, sub_path, out_file, effective_minsup_percent, keep_pattern_files, spmf_filter_count)

        recs.append({
            "algorithm": alg,
            "transaction_ratio_percent": float(r),
            "n_transactions_sub": int(n_sub),
            "minsup_percent": float(ms_default),
            "tx_sweep_minsup_mode": mode,
            "effective_minsup_percent": float(effective_minsup_percent),
            "minsup_count_threshold": int(minsup_count_threshold),
            "fixed_minsup_count": int(fixed_minsup_count) if fixed_minsup_count is not None else None,
            "base_n_tx_for_fixed_minsup": int(base_n_tx_for_fixed) if base_n_tx_for_fixed is not None else None,
            "runtime_sec": float(m["runtime_sec"]),
            "pattern_count": int(m["pattern_count"]),
            "depth_proxy": int(m["max_itemset_len"]),
            "cmd": m["cmd"],
            "pattern_files_deleted": (not keep_pattern_files),
        })

    if "CICLAD" in baselines:
        cached = _cache_lookup("CICLAD")
        if cached is not None:
            recs.append(cached)
        else:
            fci_file = os.path.join(ds_dir, f"CICLAD_tx{int(r)}_fci.txt")
            log_file = os.path.join(ds_dir, f"CICLAD_tx{int(r)}.log")

            m = run_ciclad_multi(
                input_dat=sub_path,
                fci_out_path=fci_file,
                log_path=log_file,
                nbr_items=nbr_items,
                window_size=n_sub,
                minsup_counts=[int(minsup_count_threshold)],
                keep_pattern_files=keep_pattern_files,
            )

            count = int(m["dumped_fci_by_minsup"].get(int(minsup_count_threshold), 0))
            recs.append({
                "algorithm": "CICLAD",
                "transaction_ratio_percent": float(r),
                "n_transactions_sub": int(n_sub),
                "minsup_percent": float(ms_default),
                "tx_sweep_minsup_mode": mode,
                "effective_minsup_percent": float(effective_minsup_percent),
                "minsup_count_threshold": int(minsup_count_threshold),
                "minsup_count": int(minsup_count_threshold),
                "fixed_minsup_count": int(fixed_minsup_count) if fixed_minsup_count is not None else None,
                "base_n_tx_for_fixed_minsup": int(base_n_tx_for_fixed) if base_n_tx_for_fixed is not None else None,
                "runtime_sec": float(m["runtime_sec_wall"]),
                "pattern_count": count,
                "depth_proxy": 0,
                "ciclad_log_path": os.path.basename(log_file),
                "cmd": m["cmd"],
                "pattern_files_deleted": (not keep_pattern_files),
            })

    return ds, float(r), recs


def worker_minsup_sweep(ds, spmf_path, dat_path, n_tx, nbr_items, minsup_ratios, baselines, ds_dir, resume, cache_ms, keep_pattern_files):
    """
    Run minsup sweep for one dataset:
      - SPMF runs per minsup%
      - CICLAD runs once with multiple minsup COUNTS (only if CICLAD baseline selected)
    Returns: (ds, recs[list-of-dicts])
    """
    recs = []

    for ms in minsup_ratios:
        for alg in [a for a in baselines if a in {"FPGrowth_itemsets", "Eclat", "Hamm"}]:
            key = (alg, float(ms), None)
            if resume and key in cache_ms:
                recs.append(cache_ms[key])
                continue

            out_file = os.path.join(ds_dir, f"{alg}_ms{ms}.spmf")
            if alg == "Hamm":
                m = run_hamm(spmf_path, out_file, ms, keep_pattern_files)
            else:
                m = run_spmf(alg, spmf_path, out_file, ms, keep_pattern_files=keep_pattern_files)

            recs.append({
                "algorithm": alg,
                "transaction_ratio_percent": 100.0,
                "minsup_percent": float(ms),
                "runtime_sec": float(m["runtime_sec"]),
                "pattern_count": int(m["pattern_count"]),
                "depth_proxy": int(m["max_itemset_len"]),
                "cmd": m["cmd"],
                "pattern_files_deleted": (not keep_pattern_files),
            })

    if "CICLAD" in baselines:
        ciclad_counts = [int(math.ceil(ms/100.0 * n_tx)) for ms in minsup_ratios]
        ciclad_fci = os.path.join(ds_dir, "CICLAD_minsupSweep_fci.txt")
        ciclad_log = os.path.join(ds_dir, "CICLAD_minsupSweep.log")

        need = True
        if resume:
            ok = True
            for ms, ms_count in zip(minsup_ratios, ciclad_counts):
                if ("CICLAD", float(ms), int(ms_count)) not in cache_ms:
                    ok = False
                    break
            if ok and os.path.exists(ciclad_log):
                need = False

        if need:
            m = run_ciclad_multi(
                input_dat=dat_path,
                fci_out_path=ciclad_fci,
                log_path=ciclad_log,
                nbr_items=nbr_items,
                window_size=n_tx,
                minsup_counts=ciclad_counts,
                keep_pattern_files=keep_pattern_files,
            )
            dumped_by = m["dumped_fci_by_minsup"]
            wall = float(m["runtime_sec_wall"])
            cmd = m["cmd"]
        else:
            parsed = parse_ciclad_log(ciclad_log)
            dumped_by = parsed["dumped_fci_by_minsup"]
            wall = 0.0
            cmd = f"(skipped; see {os.path.basename(ciclad_log)})"

        for ms, ms_count in zip(minsup_ratios, ciclad_counts):
            recs.append({
                "algorithm": "CICLAD",
                "transaction_ratio_percent": 100.0,
                "minsup_percent": float(ms),
                "minsup_count": int(ms_count),
                "runtime_sec": wall,
                "pattern_count": int(dumped_by.get(int(ms_count), 0)),
                "depth_proxy": 0,
                "ciclad_log_path": os.path.basename(ciclad_log),
                "cmd": cmd,
                "pattern_files_deleted": (not keep_pattern_files),
            })

    return ds, recs

# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Run FIM baselines (FP-Growth, Eclat, CICLAD)")
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS_ALL),
                        help=f"comma list (default: {','.join(DATASETS_ALL)})")
    parser.add_argument("--tx-ratios", type=str, default=",".join(map(str, DEFAULT_TX_RATIOS)),
                        help="e.g., '10,20,30,50,70,100'")
    parser.add_argument("--minsup-ratios", type=str, default=",".join(map(str, DEFAULT_MINSUP_SWEEP)),
                        help="e.g., '0.5,1,2,5,10'")
    parser.add_argument("--override-default-minsup", type=str, default="",
                        help="dataset=val, comma-separated. e.g., 'mushroom=1,connect4=1,tic-tac-toe=2,car=5,kr-vs-kp=1'")
    parser.add_argument("--tx-sweep-minsup-mode", type=str, default="percent", choices=["percent","count"],
                        help="For tx-ratio sweep: percent=fixed minsup ratio (default); count=fixed minsup COUNT computed from full dataset size (100% tx * minsup ratio).")
    parser.add_argument("--baselines", type=str, default="FPGrowth_itemsets,Eclat,CICLAD",
                        help="Comma-separated baselines to run. Any subset of: FPGrowth_itemsets,Eclat,CICLAD")
    parser.add_argument("--resume", action="store_true",
                        help="skip running points already present in metrics JSON and output files")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="rebuild transactions even if results/<ds>_transactions.spmf exists")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4)//2),
                        help="parallel workers (default: half of CPU cores)")
    parser.add_argument("--keep-pattern-files", action="store_true",
                        help="keep frequent itemset/closed-itemset output files (default: delete after parsing)")
    args = parser.parse_args()
    print(f"[ok] Using JAVA_CMD for SPMF: {JAVA_CMD}")


    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tx_ratios = [float(x) for x in args.tx_ratios.split(",") if x.strip()]
    minsup_ratios = [float(x) for x in args.minsup_ratios.split(",") if x.strip()]

    # baselines selection
    selected_baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    allowed_baselines = {"FPGrowth_itemsets", "Eclat", "CICLAD", "Hamm"}
    for b in selected_baselines:
        if b not in allowed_baselines:
            raise ValueError(f"Unknown baseline: {b}. Allowed: {sorted(allowed_baselines)}")
    if not selected_baselines:
        raise ValueError("Empty --baselines. Choose from: FPGrowth_itemsets,Eclat,CICLAD")

    # override default minsup per dataset (optional)
    ms_map = dict(DEFAULT_MINSUP)
    if args.override_default_minsup:
        for kv in args.override_default_minsup.split(","):
            if not kv.strip():
                continue
            k, v = kv.split("=")
            ms_map[k.strip()] = float(v)

    pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # 1) preprocess: build .spmf and .dat, compute nbr_items
    prep = {}  # ds -> (spmf_path, dat_path, n_tx, nbr_items)
    for ds in datasets:
        print(f"[preprocess] {ds}")
        spmf_path = os.path.join(RESULTS_DIR, f"{ds}_transactions.spmf")
        dat_path  = os.path.join(RESULTS_DIR, f"{ds}_transactions.dat")
        item2id_path = os.path.join(RESULTS_DIR, f"{ds}_item2id.json")
        meta_path = os.path.join(RESULTS_DIR, f"{ds}_meta.json")

        need = args.force_preprocess or (not os.path.exists(spmf_path)) or (not os.path.exists(dat_path)) or (not os.path.exists(item2id_path))
        if need:
            txs = LOADERS[ds]()
            item2id = build_item2id_scan_order(txs)
            write_transactions_int(txs, spmf_path, item2id)
            shutil.copyfile(spmf_path, dat_path)  # CICLAD input is identical

            n_tx = len(txs)
            max_id = max(item2id.values()) if item2id else 0
            nbr_items = max_id + 1

            with open(item2id_path, "w", encoding="utf-8") as f:
                json.dump(item2id, f, indent=2)

            meta = {
                "dataset": ds,
                "n_transactions": n_tx,
                "max_item_id": max_id,
                "nbr_items_for_ciclad": nbr_items,
                "format": "one transaction per line; space-separated positive ints; 1-based ids",
                "id_assignment": "first-seen scan order over rows then columns (token=col=value)",
                "label_included": True,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        else:
            n_tx, max_id = scan_max_id(spmf_path)
            nbr_items = max_id + 1

        prep[ds] = (spmf_path, dat_path, n_tx, nbr_items)

    # 2) run experiments (parallel)
    metrics_by_ds = {}
    cache_tx_by_ds = {}
    cache_ms_by_ds = {}

    # Load existing metrics (for --resume) and build caches
    for ds in datasets:
        spmf_path, dat_path, n_tx, nbr_items = prep[ds]
        ds_dir = os.path.join(RESULTS_DIR, ds)
        pathlib.Path(ds_dir).mkdir(parents=True, exist_ok=True)

        metrics = load_metrics_if_any(ds_dir, ds)
        metrics_by_ds[ds] = metrics
        c_tx, c_ms = build_resume_cache(metrics)
        cache_tx_by_ds[ds] = c_tx
        cache_ms_by_ds[ds] = c_ms

    keep_pattern_files = bool(args.keep_pattern_files)

    # ----------------------
    # A) transaction ratio sweep (fixed minsup%) in parallel over (dataset, tx_ratio)
    # ----------------------
    tx_tasks = []
    for ds in datasets:
        spmf_path, dat_path, n_tx, nbr_items = prep[ds]
        ds_dir = os.path.join(RESULTS_DIR, ds)
        ms_default = ms_map.get(ds, 1.0)
        for r in tx_ratios:
            tx_tasks.append((ds, r, spmf_path, nbr_items, n_tx, ms_default, args.tx_sweep_minsup_mode, selected_baselines, ds_dir, args.resume, cache_tx_by_ds[ds], keep_pattern_files))

    tx_results = {ds: {} for ds in datasets}  # ds -> r -> {alg: rec}

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(worker_txratio_point, *t) for t in tx_tasks]
        for fut in as_completed(futures):
            ds, r, recs = fut.result()

            by_alg = {rec["algorithm"]: rec for rec in recs}
            tx_results[ds][r] = by_alg

            metrics = metrics_by_ds[ds]
            for rec in recs:
                if not find_cached(metrics.get("by_txratio", []),
                                   algorithm=rec.get("algorithm"),
                                   transaction_ratio_percent=float(rec.get("transaction_ratio_percent")),
                                   minsup_percent=float(rec.get("minsup_percent")),
                                   tx_sweep_minsup_mode=rec.get("tx_sweep_minsup_mode","percent"),
                                   minsup_count_threshold=rec.get("minsup_count_threshold", None)):
                    metrics.setdefault("by_txratio", []).append(rec)

                try:
                    cache_tx_by_ds[ds][(rec.get("algorithm"), float(rec.get("transaction_ratio_percent")), float(rec.get("minsup_percent")))] = rec
                    cache_tx_by_ds[ds][(rec.get("algorithm"), float(rec.get("transaction_ratio_percent")), str(rec.get("tx_sweep_minsup_mode","percent")).lower(), float(rec.get("minsup_percent")), int(rec.get("minsup_count_threshold")) if rec.get("minsup_count_threshold", None) is not None else None)] = rec
                except Exception:
                    pass

            ds_dir = os.path.join(RESULTS_DIR, ds)
            with open(os.path.join(ds_dir, f"metrics_{ds}.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    
    # Plot tx-ratio per dataset
    for ds in datasets:
        ds_dir = os.path.join(RESULTS_DIR, ds)
        ms_default = ms_map.get(ds, 1.0)
        mode = str(args.tx_sweep_minsup_mode).lower()
    
        y_all = {alg: [] for alg in selected_baselines}
        for r in tx_ratios:
            recs = tx_results.get(ds, {}).get(float(r), {})
            for alg in selected_baselines:
                y_all[alg].append(float(recs[alg]["runtime_sec"]) if alg in recs else float("nan"))
    
        if mode == "count":
            subtitle = f"fixed minsup_count = ceil({ms_default}% * N_full)"
            out_png = os.path.join(ds_dir, f"txratio_runtime_{ds}_count.png")
        else:
            subtitle = f"fixed minsup = {ms_default}%"
            out_png = os.path.join(ds_dir, f"txratio_runtime_{ds}_percent.png")
    
        plot_multi(tx_ratios, y_all,
                   f"{ds} — runtime vs. transaction ratio ({subtitle})",
                   "transaction ratio (%)", "runtime (s)",
                   out_png)
    
    # ----------------------
    # B) minsup ratio sweep (full dataset) in parallel over dataset
    # ----------------------
    ms_tasks = []
    for ds in datasets:
        spmf_path, dat_path, n_tx, nbr_items = prep[ds]
        ds_dir = os.path.join(RESULTS_DIR, ds)
        ms_tasks.append((ds, spmf_path, dat_path, n_tx, nbr_items, minsup_ratios, selected_baselines, ds_dir, args.resume, cache_ms_by_ds[ds], keep_pattern_files))

    ms_results = {ds: {} for ds in datasets}  # ds -> (alg, ms) -> rec

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(worker_minsup_sweep, *t) for t in ms_tasks]
        for fut in as_completed(futures):
            ds, recs = fut.result()

            idx = {}
            for rec in recs:
                idx[(rec["algorithm"], float(rec["minsup_percent"]))] = rec
            ms_results[ds] = idx

            metrics = metrics_by_ds[ds]
            for rec in recs:
                alg = rec.get("algorithm")
                ms = float(rec.get("minsup_percent"))
                mc = rec.get("minsup_count", None)
                mc = int(mc) if mc is not None else None

                if not find_cached(metrics.get("by_minsup", []),
                                   algorithm=alg,
                                   transaction_ratio_percent=100.0,
                                   minsup_percent=ms,
                                   minsup_count=mc if mc is not None else None):
                    metrics.setdefault("by_minsup", []).append(rec)

                try:
                    cache_ms_by_ds[ds][(alg, ms, mc)] = rec
                except Exception:
                    pass

            ds_dir = os.path.join(RESULTS_DIR, ds)
            with open(os.path.join(ds_dir, f"metrics_{ds}.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    # Plot minsup per dataset
    for ds in datasets:
        ds_dir = os.path.join(RESULTS_DIR, ds)
        idx = ms_results.get(ds, {})

        y_all = {alg: [] for alg in selected_baselines}
        for ms in minsup_ratios:
            for alg in selected_baselines:
                rec = idx.get((alg, float(ms)))
                y_all[alg].append(float(rec["runtime_sec"]) if rec else float("nan"))

        plot_multi(minsup_ratios, y_all,
                   f"{ds} — runtime vs. minsup",
                   "minsup (%)", "runtime (s)",
                   os.path.join(ds_dir, f"minsup_runtime_{ds}.png"))

if __name__ == "__main__":
    main()