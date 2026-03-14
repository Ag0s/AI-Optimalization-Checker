#!/usr/bin/env python3
import platform
import psutil
import subprocess
import time
import os
import numpy as np
from textwrap import dedent

# -----------------------------
# Configurable settings
# -----------------------------
LLAMACPP_BINARY = "./main"          # Path to llama.cpp binary (optional)
LLAMACPP_MODEL  = "./model.gguf"    # Path to a GGUF model (optional)
RUN_LLAMACPP_BENCH = False          # Set to True if you want to run a real LLM benchmark

# -----------------------------
# Helpers
# -----------------------------
def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
    except Exception:
        return ""

def check_cpu_flags():
    flags = []
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo") as f:
                data = f.read()
            for line in data.split("\n"):
                if "flags" in line:
                    flags = line.split(":")[1].strip().split()
                    break
        except Exception:
            pass
    return flags

def detect_disk_type():
    # Linux-specific heuristic
    result = run_cmd("lsblk -d -o name,rota")
    if not result:
        return {"unknown": "unknown"}
    lines = result.strip().split("\n")[1:]
    disk_info = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        name, rota = parts
        disk_info[name] = "SSD/NVMe" if rota == "0" else "HDD"
    return disk_info or {"unknown": "unknown"}

def memory_bandwidth_test(size_mb=500):
    size = size_mb * 1024 * 1024 // 8
    a = np.random.rand(size)
    b = np.random.rand(size)
    start = time.time()
    c = a + b
    end = time.time()
    mbps = size_mb / (end - start)
    return mbps

def cpu_thread_test():
    start = time.time()
    total = 0
    for _ in range(10_000_000):
        total += 1
    end = time.time()
    return end - start

def detect_numa():
    # Linux-only simple NUMA detection
    nodes = []
    sysfs_numa = "/sys/devices/system/node"
    if os.path.isdir(sysfs_numa):
        for entry in os.listdir(sysfs_numa):
            if entry.startswith("node"):
                nodes.append(entry)
    return nodes

def llama_cpp_benchmark():
    if not RUN_LLAMACPP_BENCH:
        return None
    if not (os.path.isfile(LLAMACPP_BINARY) and os.path.isfile(LLAMACPP_MODEL)):
        return None
    cmd = f'{LLAMACPP_BINARY} -m {LLAMACPP_MODEL} -p "Hello" --n-predict 64 --timings'
    out = run_cmd(cmd)
    if not out:
        return None
    # crude parse: look for "tokens per second"
    tps = None
    for line in out.splitlines():
        if "tokens per second" in line.lower():
            # e.g. "XXX tokens per second"
            parts = line.split()
            for p in parts:
                try:
                    tps = float(p)
                    break
                except ValueError:
                    continue
    return {"raw_output": out, "tps": tps}

# -----------------------------
# Scoring logic
# -----------------------------
def score_system(cpu_flags, mem_total, bw, cpu_time, disks, llama_bench):
    """
    Return dict with scores and reasons for 7B, 13B, 14B.
    Score: 0–100
    """
    scores = {}
    reasons = {}

    avx2 = "avx2" in cpu_flags
    avx512 = "avx512f" in cpu_flags
    fast_storage = "HDD" not in disks.values()
    ram_gb = mem_total / 1e9

    # Base CPU capability score
    cpu_score = 30
    if avx512:
        cpu_score += 40
    elif avx2:
        cpu_score += 25
    else:
        cpu_score += 5

    # Single-thread performance
    if cpu_time < 1.0:
        cpu_score += 20
    elif cpu_time < 1.5:
        cpu_score += 10
    else:
        cpu_score += 0

    # Memory bandwidth contribution
    mem_score = 0
    if bw > 8000:
        mem_score += 30
    elif bw > 4000:
        mem_score += 20
    elif bw > 2000:
        mem_score += 10

    # Storage
    storage_score = 10 if fast_storage else 0

    # Llama.cpp TPS if available
    llama_score = 0
    if llama_bench and llama_bench.get("tps"):
        tps = llama_bench["tps"]
        if tps > 20:
            llama_score += 30
        elif tps > 10:
            llama_score += 20
        elif tps > 5:
            llama_score += 10

    base_total = cpu_score + mem_score + storage_score + llama_score
    base_total = max(0, min(100, base_total))

    # Now derive per-model scores with RAM constraints
    # 7B
    s7 = base_total
    r7 = []
    if ram_gb < 8:
        s7 -= 25
        r7.append("Limited RAM (<8GB) — even 7B models may require aggressive quantization (Q3_K).")
    else:
        r7.append("Sufficient RAM for 7B models with Q4_K or better.")
    if not avx2 and not avx512:
        r7.append("No AVX2/AVX512 — CPU-only inference will be slow.")
    else:
        r7.append("Vector instructions available — good fit for quantized 7B models.")
    if not fast_storage:
        r7.append("HDD storage — model load times will be slow.")
    else:
        r7.append("SSD/NVMe storage — fast model loading.")
    if bw < 3000:
        r7.append("Low memory bandwidth — keep batch size small and context modest.")
    else:
        r7.append("Decent memory bandwidth — can use larger batch sizes for throughput.")
    scores["7B"] = max(0, min(100, s7))
    reasons["7B"] = r7

    # 13B
    s13 = base_total
    r13 = []
    if ram_gb < 12:
        s13 -= 30
        r13.append("RAM <12GB — 13B models will be very constrained, likely require Q3_K and short context.")
    elif ram_gb < 16:
        s13 -= 10
        r13.append("RAM between 12–16GB — 13B possible with Q4_K and careful context/batch settings.")
    else:
        r13.append("RAM ≥16GB — 13B models in Q4_K are realistic.")
    if not avx2 and not avx512:
        s13 -= 10
        r13.append("No AVX2/AVX512 — 13B on CPU will be quite slow.")
    else:
        r13.append("Vector instructions present — 13B feasible with quantization.")
    if bw < 4000:
        s13 -= 10
        r13.append("Moderate/low memory bandwidth — 13B will be sensitive to batch size and context length.")
    else:
        r13.append("Good memory bandwidth — 13B throughput acceptable with tuning.")
    scores["13B"] = max(0, min(100, s13))
    reasons["13B"] = r13

    # 14B (similar to 13B but slightly stricter)
    s14 = base_total
    r14 = []
    if ram_gb < 16:
        s14 -= 30
        r14.append("RAM <16GB — 14B models will be heavily constrained, require strong quantization and short context.")
    else:
        r14.append("RAM ≥16GB — 14B models possible with Q4_K and careful tuning.")
    if not avx2 and not avx512:
        s14 -= 15
        r14.append("No AVX2/AVX512 — 14B on CPU will be very slow.")
    else:
        r14.append("Vector instructions present — 14B feasible but slower than 7B/13B.")
    if bw < 5000:
        s14 -= 10
        r14.append("Memory bandwidth not ideal — 14B will be sensitive to batch size and context length.")
    else:
        r14.append("Good memory bandwidth — 14B can be usable with proper tuning.")
    scores["14B"] = max(0, min(100, s14))
    reasons["14B"] = r14

    return scores, reasons

# -----------------------------
# Main
# -----------------------------
def main():
    # Collect system info
    cpu_model = platform.processor()
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    flags = check_cpu_flags()
    avx2 = "avx2" in flags
    avx512 = "avx512f" in flags

    mem = psutil.virtual_memory()
    disks = detect_disk_type()
    numa_nodes = detect_numa()

    bw = memory_bandwidth_test()
    cpu_time = cpu_thread_test()
    llama_bench = llama_cpp_benchmark()

    scores, reasons = score_system(flags, mem.total, bw, cpu_time, disks, llama_bench)

    # Build Markdown report
    report = []

    report.append("# LLM CPU Optimization Diagnostic Report\n")

    report.append("## 1. System Overview\n")
    report.append(f"- **CPU Model:** {cpu_model or 'Unknown'}")
    report.append(f"- **Physical Cores:** {physical}")
    report.append(f"- **Logical Threads:** {logical}")
    report.append(f"- **AVX2 Support:** {'Yes' if avx2 else 'No'}")
    report.append(f"- **AVX512 Support:** {'Yes' if avx512 else 'No'}")
    report.append(f"- **Total RAM:** {round(mem.total / 1e9, 2)} GB")
    report.append(f"- **NUMA Nodes Detected:** {', '.join(numa_nodes) if numa_nodes else 'None / Not detected'}")

    report.append("\n### Disk Devices\n")
    for d, t in disks.items():
        report.append(f"- **{d}:** {t}")

    report.append("\n## 2. Synthetic Benchmarks\n")
    report.append(f"- **Estimated Memory Bandwidth:** {bw:.2f} MB/s")
    report.append(f"- **Single-thread Loop Time (10M ops):** {cpu_time:.2f} s")

    if llama_bench:
        report.append("\n### llama.cpp Benchmark\n")
        tps = llama_bench.get("tps")
        if tps:
            report.append(f"- **Tokens per second (approx):** {tps:.2f}")
        else:
            report.append("- **Tokens per second:** Not parsed, see raw output.")
        report.append("\n<details>\n<summary>Raw llama.cpp output</summary>\n\n```text\n")
        report.append(llama_bench["raw_output"])
        report.append("\n```\n</details>\n")
    else:
        report.append("\n> llama.cpp benchmark not run or not configured. Set `RUN_LLAMACPP_BENCH = True` and adjust paths if you want this.\n")

    report.append("\n## 3. Model Suitability Scores\n")
    for model in ["7B", "13B", "14B"]:
        report.append(f"### {model} Models")
        report.append(f"- **Score:** {scores[model]}/100")
        report.append("- **Reasons:**")
        for r in reasons[model]:
            report.append(f"  - {r}")
        report.append("")

    report.append("## 4. Optimization Targets and What to Look At\n")

    report.append("### 4.1 CPU & Instruction Set\n")
    report.append("- **Why it matters:** LLM runtimes rely heavily on vector instructions (AVX2/AVX512) for matrix multiplications.")
    if avx512:
        report.append("- **Status:** AVX512 available → ideal for CPU LLMs.")
        report.append("- **Action:** Use GGUF models with `llama.cpp`, prefer Q4_K_M or Q5_K quantization.")
    elif avx2:
        report.append("- **Status:** AVX2 available → good performance.")
        report.append("- **Action:** Use Q4_K_M for quality, Q3_K for speed. Stick to llama.cpp/Ollama for best CPU kernels.")
    else:
        report.append("- **Status:** No AVX2/AVX512 detected.")
        report.append("- **Action:** Consider smaller models (3B–7B) and strong quantization (Q3_K). GPU offload is recommended if possible.")

    report.append("\n### 4.2 RAM Capacity & Bandwidth\n")
    report.append("- **Why it matters:** LLMs are memory-bandwidth bound; RAM size limits model size and context length.")
    if mem.total < 12e9:
        report.append("- **Status:** RAM is limited (<12GB).")
        report.append("- **Action:** Prefer 7B models in Q3_K/Q4_K, keep context length around 2048, avoid large batch sizes.")
    else:
        report.append("- **Status:** RAM is sufficient for 13B+ models.")
        report.append("- **Action:** 13B/14B in Q4_K_M are realistic; still keep an eye on context length and batch size.")

    if bw < 3000:
        report.append(f"- **Bandwidth:** {bw:.2f} MB/s → low.")
        report.append("- **Action:**")
        report.append("  - Ensure RAM is running in dual-channel mode (check BIOS and physical DIMM placement).")
        report.append("  - Use smaller batch sizes (e.g., 128–256).")
        report.append("  - Avoid very long context windows unless necessary.")
    elif bw < 6000:
        report.append(f"- **Bandwidth:** {bw:.2f} MB/s → moderate.")
        report.append("- **Action:** Batch sizes of 256–384 are usually safe; test and adjust.")
    else:
        report.append(f"- **Bandwidth:** {bw:.2f} MB/s → good.")
        report.append("- **Action:** You can push batch sizes to 384–512 for better throughput.")

    report.append("\n### 4.3 Storage\n")
    report.append("- **Why it matters:** Storage mainly affects model load time, not tokens-per-second.")
    if "HDD" in disks.values():
        report.append("- **Status:** HDD detected.")
        report.append("- **Action:** Move model files to SSD/NVMe. This will significantly reduce startup time.")
    else:
        report.append("- **Status:** SSD/NVMe detected.")
        report.append("- **Action:** No major changes needed; loading should be fast.")

    report.append("\n### 4.4 NUMA Topology\n")
    if numa_nodes:
        report.append("- **Status:** Multiple NUMA nodes detected.")
        report.append("- **Why it matters:** Cross-node memory access is slower; pinning threads and memory to a single node can improve performance.")
        report.append("- **Action:**")
        report.append("  - Use `numactl --cpunodebind=0 --membind=0` (or similar) when launching llama.cpp.")
        report.append("  - Keep the LLM process on a single NUMA node if possible.")
    else:
        report.append("- **Status:** No NUMA nodes detected or single-node system.")
        report.append("- **Action:** No NUMA-specific tuning required.")

    report.append("\n### 4.5 OS & Power Settings\n")
    report.append("- **Why it matters:** Power-saving modes can throttle CPU frequency and hurt LLM throughput.")
    report.append("- **Linux:**")
    report.append("  - Set CPU governor to performance:")
    report.append("    ```bash\n    sudo cpupower frequency-set -g performance\n    ```")
    report.append("- **Windows:**")
    report.append("  - Use 'High Performance' or 'Ultimate Performance' power plan.")
    report.append("  - Disable aggressive CPU power saving in advanced power settings.")

    report.append("\n### 4.6 LLM Runtime Parameters\n")
    report.append("- **Threads:** Set to the number of physical cores (not logical threads).")
    report.append("- **Batch size:**")
    report.append("  - Low bandwidth → 128–256")
    report.append("  - Moderate bandwidth → 256–384")
    report.append("  - High bandwidth → 384–512")
    report.append("- **Context length:**")
    report.append("  - Use 2048–4096 unless you truly need more; longer contexts increase memory pressure and latency.")
    report.append("- **Quantization:**")
    report.append("  - For speed: Q3_K / Q4_K")
    report.append("  - For quality: Q4_K_M / Q5_K (if CPU and RAM allow)")
    report.append("- **Preferred runtimes:** llama.cpp, Ollama, LM Studio (for GUI).")

    report.append("\n---\n")
    report.append("This report is a starting point. You can now:\n")
    report.append("- Adjust quantization and model size based on the scores.\n")
    report.append("- Tune batch size and context length according to memory bandwidth.\n")
    report.append("- Apply BIOS/OS tweaks to unlock more performance.\n")

    # Print report
    print("\n".join(report))


if __name__ == "__main__":
    main()
