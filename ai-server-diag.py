#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Server Diagnostics & Model Recommendation Tool
Optimized for CPU inference, GPU-ready for future upgrades.

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

# -----------------------------
# Utility helpers
# -----------------------------

def run(cmd, timeout=5):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True).strip()
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

def get_cpu_info():
    info = {
        "model": None,
        "vendor": None,
        "cores": os.cpu_count(),
        "physical": None,
        "flags": [],
        "l3_kb": None,
    }

    out = run(["lscpu"])
    if out:
        kv = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in out.splitlines() if ":" in line)}
        info["model"] = kv.get("Model name")
        info["vendor"] = kv.get("Vendor ID")
        info["physical"] = int(kv.get("Core(s) per socket", "0")) * int(kv.get("Socket(s)", "1"))
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

    return {"nodes": len(nodes), "detail": nodes}

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

    out = run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        name, mem, cc = [x.strip() for x in line.split(",")]
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

# -----------------------------
# Model recommendation engine
# -----------------------------

def estimate_tokens_per_sec(cpu, mem_bw, caps):
    """
    Very rough estimator for CPU-only LLM inference.
    """
    base = 1.0

    # Core scaling
    base *= cpu["cores"] / 8

    # AVX2 / AVX512
    if caps["avx512"]:
        base *= 2.2
    elif caps["avx2"]:
        base *= 1.4
    else:
        base *= 0.5

    # Memory bandwidth
    if mem_bw:
        base *= min(mem_bw / 15000, 1.0)

    return max(base, 0.1)

def recommend_model(cpu, mem, caps, mem_bw):
    """
    Returns recommended model size for balanced quality/performance.
    """
    ram_gb = mem["total"] / (1024**3)
    tps = estimate_tokens_per_sec(cpu, mem_bw, caps)

    # Model RAM requirements (Q4_K_M)
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

    # Choose based on performance
    if tps < 0.8:
        return feasible[0], tps
    if tps < 1.5:
        return feasible[min(1, len(feasible)-1)], tps
    if tps < 2.5:
        return feasible[min(2, len(feasible)-1)], tps

    return feasible[-1], tps

# -----------------------------
# Tuning script
# -----------------------------

def tuning_script():
    return """#!/usr/bin/env bash
# AI Server Tuning Script (safe defaults)
set -e

echo "Setting Transparent Huge Pages to madvise..."
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled || true

echo "Setting swappiness to 10..."
sysctl -w vm.swappiness=10

echo "Setting CPU governor to performance..."
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  if [ -f "$c/cpufreq/scaling_governor" ]; then
    echo performance > "$c/cpufreq/scaling_governor"
  fi
done

echo "Done. Reboot recommended."
"""

# -----------------------------
# Report generation
# -----------------------------

def generate_report(cpu, mem, numa, gpu, caps, pybench, mbw, model, tps):
    lines = []
    lines.append("# AI Server Diagnostic Report")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")

    lines.append("## CPU")
    lines.append(f"- Model: {cpu['model']}")
    lines.append(f"- Cores: {cpu['cores']}")
    lines.append(f"- L3 Cache: {cpu['l3_kb']}")
    lines.append(f"- AVX2: {caps['avx2']}")
    lines.append(f"- AVX-512: {caps['avx512']}")
    lines.append("")

    lines.append("## Memory")
    lines.append(f"- Total RAM: {human(mem['total'])}")
    lines.append(f"- Available: {human(mem['avail'])}")
    lines.append("")

    lines.append("## NUMA")
    lines.append(f"- Nodes: {numa['nodes']}")
    for n in numa["detail"]:
        lines.append(f"  - Node {n['id']}: CPUs {n['cpus']}, Mem {human(n['mem_kb']*1024)}")
    lines.append("")

    lines.append("## Benchmarks")
    lines.append(f"- Python CPU ops/sec: {pybench:,.0f}")
    lines.append(f"- Memory bandwidth (mbw): {mbw or 'N/A'} MB/s")
    lines.append("")

    lines.append("## Model Recommendation")
    lines.append(f"- Recommended model: **{model}**")
    lines.append(f"- Estimated tokens/sec: **{tps:.2f}**")
    lines.append("")

    lines.append("## GPU")
    if gpu["has_gpu"]:
        for g in gpu["gpus"]:
            lines.append(f"- {g['name']} ({g['mem_mib']} MiB)")
    else:
        lines.append("- No NVIDIA GPU detected")
    lines.append("")

    lines.append("## Tuning Suggestions")
    lines.append("- Set THP to `madvise`")
    lines.append("- Set CPU governor to `performance`")
    lines.append("- Set swappiness to 10")
    lines.append("- Pin inference to a single NUMA node")
    lines.append("")

    return "\n".join(lines)

# -----------------------------
# Main
# -----------------------------

def main():
    cpu = get_cpu_info()
    mem = get_memory()
    numa = get_numa()
    gpu = get_gpu()
    caps_info = cpu_caps(cpu)

    pybench = bench_python_cpu()
    mbw = bench_mbw()

    model, tps = recommend_model(cpu, mem, caps_info, mbw)

    report = generate_report(cpu, mem, numa, gpu, caps_info, pybench, mbw, model, tps)
    Path("ai-server-report.md").write_text(report)
    Path("ai-server-tune.sh").write_text(tuning_script())
    Path("ai-server-tune.sh").chmod(0o755)

    print("Report written to ai-server-report.md")
    print("Tuning script written to ai-server-tune.sh")

if __name__ == "__main__":
    main()
