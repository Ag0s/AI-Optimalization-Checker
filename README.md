# AI Server Diagnostic & Optimization Tool — Technical Specification

## 1. Purpose
A Linux-based diagnostic tool designed to evaluate and optimize servers for **CPU‑based LLM inference**, while remaining **GPU‑ready** for future upgrades.  
It analyzes hardware, benchmarks performance, recommends optimal model sizes, and generates NUMA‑aware execution strategies.

---

## 2. Core Capabilities

### 2.1 Hardware & System Detection
The tool collects:
- CPU model, vendor, logical/physical cores  
- CPU ISA support: AVX2, AVX‑512, VNNI, BF16, FMA  
- L3 cache size  
- NUMA topology (nodes, CPU lists, memory per node)  
- Total/available RAM, swap usage  
- NVIDIA GPU detection (optional)  
- OS, kernel, Python version  

---

### 2.2 Benchmarks
Lightweight performance tests:
- **Python loop benchmark** — raw interpreter throughput  
- **Memory bandwidth** — via `mbw` if installed  
- **Matmul GFLOP/s** — NumPy accelerated if available  

These metrics feed into the performance estimator.

---

## 3. Model Recommendation Engine

### 3.1 Inputs
- RAM capacity  
- CPU ISA capabilities  
- Core count  
- Memory bandwidth  
- Matmul GFLOP/s  

### 3.2 Outputs
- Recommended LLM model size (3B, 7B, 13B, 30B, 70B)  
- Estimated tokens/sec for CPU-only inference  

### 3.3 Assumptions
Approximate RAM requirements (Q4_K_M quantization):
- 3B → 6 GB  
- 7B → 12 GB  
- 13B → 24 GB  
- 30B → 48 GB  
- 70B → 96 GB  

---

## 4. NUMA Intelligence

### 4.1 NUMA Node Scoring
Each node is evaluated using:
- Local memory size  
- Number of CPU cores  
- L3 cache per core  

Highest‑scoring node is selected for inference.

### 4.2 NUMA Health Score
A 0–100 metric indicating system balance:
- 100 = single node or perfectly symmetric  
- Lower scores indicate memory/core imbalance  

### 4.3 NUMA‑Aware CPU Pinning Generator
Produces:
- `numactl` binding  
- `taskset` CPU mask  
- Recommended thread count  

Example:
`numactl --cpunodebind=1 --membind=1 taskset -c 16-31 ./llama-cli -m model.gguf -t 16`


### 4.4 Multi‑NUMA Parallel Inference Mode
For multi-socket systems, generates one command per NUMA node to maximize throughput.

---

## 5. System Optimization Recommendations

### 5.1 Auto‑Generated Tuning Script
Creates `ai-server-tune.sh` with safe defaults:
- Transparent Huge Pages → `madvise`  
- CPU governor → `performance`  
- Swappiness → `10`  
- No dangerous GRUB edits  
- irqbalance preserved  

---

## 6. Output Files

### 6.1 ai-server-report.md
Includes:
- System overview  
- CPU capabilities  
- Memory & NUMA topology  
- Benchmarks  
- Model recommendation  
- NUMA pinning commands  
- Multi‑NUMA inference suggestions  
- Tuning recommendations  

### 6.2 ai-server-tune.sh
Safe, review-before-use optimization script.

---

## 7. Target Users
- AI researchers  
- Self-hosters  
- Edge inference operators  
- System administrators  
- Anyone optimizing CPU-based LLM inference  

---

## 8. Supported Environments
- Linux (Ubuntu, Debian, RHEL, Rocky, Alma, Arch, etc.)  
- x86‑64 CPUs (Intel & AMD)  
- Optional NVIDIA GPUs  

---

## 9. Design Goals
- Lightweight and dependency-minimal  
- Safe system tuning  
- Accurate CPU inference estimation  
- Extensible for GPU and distributed inference  
