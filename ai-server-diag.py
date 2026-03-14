#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Server Diagnostics & Model Recommendation Tool
- CPU-first, GPU-ready
- NUMA-aware pinning generator
- NUMA health score
- Multi-NUMA parallel inference suggestions
- Lightweight benchmarks (Python loop, mbw, matmul)

Outputs:
- ai-server-report.md
- ai-server-tune.sh
"""

import os
import re
import json
import time
import shutil
import socket
import platform
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import numpy as np
except ImportError:
    np = None

# -----------------------------
# Utility helpers
# -----------------------------

def run(cmd, timeout=5):
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True
        ).strip()
    except Exception:
        return ""

def read(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""

def which(cmd):
    return shutil.which(cmd) is not None

def human(n):
    if n is None:
        return "unknown"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"

# -----------------------------
# System info
# -----------------------------

def get_basic_info():
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }

def get_cpu_info():
    info = {
        "model": None,
        "vendor": None,
        "cores": os.cpu_count() or 1,
        "physical": None,
        "flags": [],
        "l3_kb": None,
    }

    out = run(["lscpu"])
    if out:
        kv = {
            k.strip(): v.strip()
            for k, v in (
                line.split(":", 1)
                for line in out.splitlines()
                if ":" in line
            )
        }
        info["model"] = kv.get("Model name")
        info["vendor"] = kv.get("Vendor ID")
        try:
            info["physical"] = int(kv.get("Core(s) per socket", "0")) * int(
                kv.get("Socket(s)", "1")
            )
        except Exception:
            info["physical"] = None
        info["flags"] = kv.get("Flags", "").split()
        info["l3_kb"] = kv.get("L3 cache")

    return info

def get_numa():
    nodes = []
    base = Path("/sys/devices/system/node")
    if not base.exists():
        return {"nodes": 1, "detail": []}

    for nd in sorted(base.glob("node[0-9]*")):
        nid = int(nd.name.replace("node", ""))
        cpus = read(nd / "cpulist")
        meminfo = read(nd / "meminfo")
        mem_kb = None
        for line in meminfo.splitlines():
            if line.startswith("MemTotal"):
                mem_kb = int(line.split()[3])
        nodes.append({"id": nid, "cpus": cpus, "mem_kb": mem_kb})

    return {"nodes": len(nodes) or 1, "detail": nodes}

def get_memory():
    if psutil:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return {
            "total": vm.total,
            "avail": vm.available,
            "swap_total": sw.total,
            "swap_used": sw.used,
        }
    return {"total": None, "avail": None, "swap_total": None, "swap_used": None}

def get_gpu():
    if not which("nvidia-smi"):
        return {"has_gpu": False, "gpus": []}
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    gpus = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            name, mem, cc = parts[:3]
            gpus.append({"name": name, "mem_mib": int(mem), "cc": cc})
    return {"has_gpu": True, "gpus": gpus}

# -----------------------------
# CPU capability analysis
# -----------------------------

def cpu_caps(cpu):
    flags = cpu["flags"]
    return {
        "avx2": "avx2" in flags,
        "avx512": any(f.startswith("avx512") for f in flags),
        "fma": "fma" in flags,
        "vnni": any("vnni" in f for f in flags),
        "bf16": any("bf16" in f for f in flags),
    }

# -----------------------------
# Benchmarks
# -----------------------------

def bench_python_cpu(sec=1.5):
    start = time.time()
    n = 0
    while time.time() - start < sec:
        n += 1
    return n / sec

def bench_mbw():
    if not which("mbw"):
        return None
    out = run(["mbw", "-n", "3", "256"], timeout=20)
    for line in reversed(out.splitlines()):
        if "MB/s" in line:
            m = re.search(r"([0-9.]+)\s+MB/s", line)
            if m:
                return float(m.group(1))
    return None

def bench_matmul():
    """
    Simple matmul benchmark.
    If NumPy is available, use it; otherwise, a tiny pure-Python fallback.
    Returns approx GFLOP/s.
    """
    size = 512
    if np is None:
        # Very small fallback to avoid huge runtimes
        size = 128
        a = [[1.0] * size for _ in range(size)]
        b = [[1.0] * size for _ in range(size)]
        start = time.time()
        c = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for k in range(size):
                aik = a[i][k]
                for j in range(size):
                    c[i][j] += aik * b[k][j]
        elapsed = time.time() - start
    else:
        a = np.random.rand(size, size).astype("float32")
        b = np.random.rand(size, size).astype("float32")
        start = time.time()
        c = a @ b
        elapsed = time.time() - start

    flops = 2 * (size**3)
    gflops = flops / elapsed / 1e9
    return gflops

# -----------------------------
# Model recommendation engine
# -----------------------------

def estimate_tokens_per_sec(cpu, mem_bw, caps_info, gflops):
    """
    Very rough estimator for CPU-only LLM inference.
    Combines cores, ISA, memory bandwidth, and matmul GFLOP/s.
    """
    base = 1.0

    # Core scaling
    base *= cpu["cores"] / 8

    # ISA
    if caps_info["avx512"]:
        base *= 2.2
    elif caps_info["avx2"]:
        base *= 1.4
    else:
        base *= 0.5

    # Memory bandwidth
    if mem_bw:
        base *= min(mem_bw / 15000, 1.0)

    # Matmul
    if gflops:
        base *= min(gflops / 200.0, 1.0)

    return max(base, 0.1)

def recommend_model(cpu, mem, caps_info, mem_bw, gflops):
    """
    Returns recommended model size for balanced quality/performance.
    """
    if not mem["total"]:
        return "unknown", 0.0

    ram_gb = mem["total"] / (1024**3)
    tps = estimate_tokens_per_sec(cpu, mem_bw, caps_info, gflops)

    # Approx RAM requirements (Q4-ish)
    req = {
        "3B": 6,
        "7B": 12,
        "13B": 24,
        "30B": 48,
        "70B": 96,
    }

    feasible = [m for m, r in req.items() if ram_gb >= r]
    if not feasible:
        return "3B", tps

    if tps < 0.8:
        return feasible[0], tps
    if tps < 1.5:
        return feasible[min(1, len(feasible) - 1)], tps
    if tps < 2.5:
        return feasible[min(2, len(feasible) - 1)], tps

    return feasible[-1], tps

# -----------------------------
# NUMA helpers
# -----------------------------

def parse_cpu_list(cpulist):
    """Convert cpulist like '0-7,16-23' into a list of ints."""
    cpus = []
    if not cpulist:
        return cpus
    for part in cpulist.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(cpus)

def score_numa_nodes(numa, cpu_info):
    """
    Score NUMA nodes based on:
    - Local memory
    - Number of cores
    - L3 cache per core (if available)
    """
    results = []

    l3_kb = None
    if cpu_info.get("l3_kb"):
        try:
            l3_kb = int(
                cpu_info["l3_kb"]
                .replace("K", "")
                .replace("KB", "")
                .replace("kB", "")
                .strip()
            )
        except Exception:
            pass

    for node in numa["detail"]:
        cpus = parse_cpu_list(node["cpus"])
        core_count = len(cpus)
        mem_gb = (node["mem_kb"] or 0) / (1024**2)

        score = core_count * 1.0 + mem_gb * 0.5
        if l3_kb and core_count > 0:
            score += (l3_kb / 1024) / core_count

        results.append(
            {
                "node": node["id"],
                "cpus": cpus,
                "core_count": core_count,
                "mem_gb": mem_gb,
                "score": score,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def numa_health_score(numa):
    """
    Simple NUMA health score:
    - 100: single node or perfectly balanced
    - lower if nodes are very imbalanced
    """
    if numa["nodes"] <= 1 or not numa["detail"]:
        return 100

    mems = [(n["mem_kb"] or 0) for n in numa["detail"]]
    cores = [len(parse_cpu_list(n["cpus"])) for n in numa["detail"]]

    if not mems or not cores:
        return 70

    mem_min, mem_max = min(mems), max(mems)
    core_min, core_max = min(cores), max(cores)

    mem_ratio = mem_min / mem_max if mem_max else 1.0
    core_ratio = core_min / core_max if core_max else 1.0

    score = 50 + 25 * mem_ratio + 25 * core_ratio
    return int(max(0, min(100, score)))

def generate_numa_pinning(best_node):
    """
    Produce:
    - numactl command
    - taskset mask
    - recommended thread count
    """
    cpus = best_node["cpus"]
    if not cpus:
        return None

    cpu_range = f"{cpus[0]}-{cpus[-1]}"
    threads = len(cpus)

    numactl_cmd = (
        f"numactl --cpunodebind={best_node['node']} "
        f"--membind={best_node['node']}"
    )
    taskset_cmd = f"taskset -c {cpu_range}"

    example_llama = (
        f"{numactl_cmd} {taskset_cmd} "
        f"./llama-cli -m model.gguf -t {threads}"
    )

    return {
        "node": best_node["node"],
        "cpu_range": cpu_range,
        "threads": threads,
        "numactl": numactl_cmd,
        "taskset": taskset_cmd,
        "example": example_llama,
    }

def generate_multi_numa_examples(numa, cpu_info):
    """
    For multi-NUMA systems, suggest running one process per node.
    """
    if numa["nodes"] <= 1:
        return []

    scored = score_numa_nodes(numa, cpu_info)
    examples = []
    for node in scored:
        pin = generate_numa_pinning(node)
        if not pin:
            continue
        cmd = (
            f"{pin['numactl']} {pin['taskset']} "
            f"./llama-cli -m model.gguf -t {pin['threads']} "
            f"# Node {pin['node']}"
        )
        examples.append(cmd)
    return examples

# -----------------------------
# Tuning script
# -----------------------------

def tuning_script():
    return """#!/usr/bin/env bash
# AI Server Tuning Script (safe defaults)
set -e

echo "Setting Transparent Huge Pages to madvise..."
if [ -d /sys/kernel/mm/transparent_hugepage ]; then
  echo madvise > /sys/kernel/mm/transparent_hugepage/enabled || true
  echo madvise > /sys/kernel/mm/transparent_hugepage/defrag || true
fi

echo "Setting swappiness to 10..."
sysctl -w vm.swappiness=10 || true

echo "Setting CPU governor to performance..."
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  if [ -f "$c/cpufreq/scaling_governor" ]; then
    echo performance > "$c/cpufreq/scaling_governor" || true
  fi
done

echo "Done. Reboot recommended."
"""

# -----------------------------
# Report generation
# -----------------------------

def generate_report(
    basic,
    cpu,
    mem,
    numa,
    gpu,
    caps_info,
    pybench,
    mbw,
    gflops,
    model,
    tps,
):
    lines = []
    lines.append("# AI Server Diagnostic Report")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")

    lines.append("## System")
    lines.append(f"- Hostname: `{basic['hostname']}`")
    lines.append(f"- OS: `{basic['os']}`")
    lines.append(f"- Kernel: `{basic['kernel']}`")
    lines.append(f"- Python: `{basic['python']}`")
    lines.append("")

    lines.append("## CPU")
    lines.append(f"- Model: `{cpu['model']}`")
    lines.append(f"- Vendor: `{cpu['vendor']}`")
    lines.append(f"- Logical cores: `{cpu['cores']}`")
    lines.append(f"- Physical cores (est.): `{cpu['physical']}`")
    lines.append(f"- L3 cache: `{cpu['l3_kb']}`")
    lines.append(f"- AVX2: `{caps_info['avx2']}`")
    lines.append(f"- AVX-512: `{caps_info['avx512']}`")
    lines.append(f"- VNNI: `{caps_info['vnni']}`")
    lines.append(f"- BF16: `{caps_info['bf16']}`")
    lines.append("")

    lines.append("## Memory")
    lines.append(f"- Total RAM: `{human(mem['total'])}`")
    lines.append(f"- Available RAM: `{human(mem['avail'])}`")
    lines.append(f"- Swap total: `{human(mem['swap_total'])}`")
    lines.append(f"- Swap used: `{human(mem['swap_used'])}`")
    lines.append("")

    lines.append("## NUMA topology")
    lines.append(f"- NUMA nodes: `{numa['nodes']}`")
    for n in numa["detail"]:
        lines.append(
            f"  - Node {n['id']}: CPUs `{n['cpus']}`, Mem `{human((n['mem_kb'] or 0)*1024)}`"
        )
    lines.append(f"- NUMA health score: `{numa_health_score(numa)}` / 100")
    lines.append("")

    lines.append("## Benchmarks")
    lines.append(f"- Python CPU ops/sec: `{pybench:,.0f}`")
    lines.append(f"- Memory bandwidth (mbw): `{mbw or 'N/A'}` MB/s")
    lines.append(f"- Matmul performance: `{gflops:.2f}` GFLOP/s")
    lines.append("")

    lines.append("## Model recommendation (CPU-only, Q4-ish)")
    lines.append(f"- Recommended model: **{model}**")
    lines.append(f"- Estimated tokens/sec: **{tps:.2f}**")
    lines.append("")

    lines.append("## GPU")
    if gpu["has_gpu"]:
        for g in gpu["gpus"]:
            lines.append(
                f"- `{g['name']}` ({g['mem_mib']} MiB, CC {g['cc']})"
            )
    else:
        lines.append("- No NVIDIA GPU detected")
    lines.append("")

    # NUMA-aware pinning
    if numa["nodes"] > 0 and numa["detail"]:
        lines.append("## NUMA-aware CPU pinning recommendation")
        best_nodes = score_numa_nodes(numa, cpu)
        best = best_nodes[0]
        pin = generate_numa_pinning(best)
        if pin:
            lines.append(f"- Best NUMA node: **{pin['node']}**")
            lines.append(f"- CPU range: `{pin['cpu_range']}`")
            lines.append(f"- Recommended threads: `{pin['threads']}`")
            lines.append("")
            lines.append("### Suggested single-process command")
            lines.append("```bash")
            lines.append(pin["example"])
            lines.append("```")
            lines.append("")
    else:
        lines.append("## NUMA-aware CPU pinning recommendation")
        lines.append("- Single-node system; NUMA pinning not required.")
        lines.append("")

    # Multi-NUMA parallel inference
    if numa["nodes"] > 1:
        lines.append("## Multi-NUMA parallel inference (advanced)")
        lines.append(
            "Run one LLM process per NUMA node for higher throughput (at the cost of more RAM)."
        )
        examples = generate_multi_numa_examples(numa, cpu)
        if examples:
            lines.append("### Example commands (one per node)")
            lines.append("```bash")
            for cmd in examples:
                lines.append(cmd)
            lines.append("```")
        lines.append("")
    else:
        lines.append("## Multi-NUMA parallel inference")
        lines.append("- Not applicable (single NUMA node).")
        lines.append("")

    lines.append("## Tuning suggestions")
    lines.append("- Set Transparent Huge Pages to `madvise`.")
    lines.append("- Set CPU governor to `performance` on dedicated inference nodes.")
    lines.append("- Set `vm.swappiness` to around 10.")
    lines.append("- Pin inference to a single NUMA node for best locality.")
    lines.append("- Use a real LLM benchmark (e.g., llama.cpp tokens/sec) to validate these estimates.")
    lines.append("")

    return "\n".join(lines)

# -----------------------------
# Main
# -----------------------------

def main():
    basic = get_basic_info()
    cpu = get_cpu_info()
    mem = get_memory()
    numa = get_numa()
    gpu = get_gpu()
    caps_info = cpu_caps(cpu)

    pybench = bench_python_cpu()
    mbw = bench_mbw()
    gflops = bench_matmul()

    model, tps = recommend_model(cpu, mem, caps_info, mbw, gflops)

    report = generate_report(
        basic,
        cpu,
        mem,
        numa,
        gpu,
        caps_info,
        pybench,
        mbw,
        gflops,
        model,
        tps,
    )

    Path("ai-server-report.md").write_text(report)
    Path("ai-server-tune.sh").write_text(tuning_script())
    Path("ai-server-tune.sh").chmod(0o755)

    print("Report written to ai-server-report.md")
    print("Tuning script written to ai-server-tune.sh")

if __name__ == "__main__":
    main()
