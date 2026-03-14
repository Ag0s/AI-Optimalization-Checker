#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Server Diagnostics & Optimization Advisor

Goals:
- Inspect CPU, memory, NUMA, storage, kernel, and power settings
- Detect AI-relevant capabilities (AVX2/AVX-512, VNNI, BF16, cache sizes)
- Run light benchmarks (optionally using external tools if present)
- Estimate suitability for common LLM sizes on CPU
- Generate a Markdown report and a conservative tuning script

This script is intentionally read-friendly and modular.
"""

import os
import re
import sys
import json
import time
import shutil
import socket
import textwrap
import platform
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

# -----------------------------
# Helpers
# -----------------------------

def run_cmd(cmd, timeout=5):
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True
        )
        return out.strip()
    except Exception:
        return ""


def which(cmd):
    return shutil.which(cmd) is not None


def read_file(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def parse_key_value_lines(text, sep=":"):
    result = {}
    for line in text.splitlines():
        if sep in line:
            k, v = line.split(sep, 1)
            result[k.strip()] = v.strip()
    return result


def human_bytes(n):
    if n is None:
        return "unknown"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


# -----------------------------
# System information
# -----------------------------

def get_basic_system_info():
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }


def get_cpu_info():
    info = {
        "model_name": None,
        "vendor": None,
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "max_freq_mhz": None,
        "flags": [],
        "l1d_kb": None,
        "l1i_kb": None,
        "l2_kb": None,
        "l3_kb": None,
        "microarch": None,
    }

    lscpu_out = run_cmd(["lscpu"])
    if lscpu_out:
        kv = parse_key_value_lines(lscpu_out)
        info["model_name"] = kv.get("Model name")
        info["vendor"] = kv.get("Vendor ID")
        try:
            info["physical_cores"] = int(kv.get("Core(s) per socket", "0")) * int(
                kv.get("Socket(s)", "1")
            )
        except ValueError:
            info["physical_cores"] = None

        info["max_freq_mhz"] = kv.get("CPU max MHz")
        flags = kv.get("Flags", "")
        info["flags"] = flags.split()

        # Caches
        info["l1d_kb"] = kv.get("L1d cache")
        info["l1i_kb"] = kv.get("L1i cache")
        info["l2_kb"] = kv.get("L2 cache")
        info["l3_kb"] = kv.get("L3 cache")

        # Microarchitecture (best-effort heuristic)
        info["microarch"] = kv.get("Model name")

    # Fallback: /proc/cpuinfo
    if not info["model_name"]:
        cpuinfo = read_file("/proc/cpuinfo")
        for block in cpuinfo.split("\n\n"):
            if "model name" in block:
                for line in block.splitlines():
                    if line.startswith("model name"):
                        info["model_name"] = line.split(":", 1)[1].strip()
                    if line.startswith("vendor_id"):
                        info["vendor"] = line.split(":", 1)[1].strip()
                break

    return info


def get_numa_info():
    nodes = []
    sysfs = Path("/sys/devices/system/node")
    if not sysfs.exists():
        return {"numa_nodes": 1, "nodes": nodes}

    for node_dir in sorted(sysfs.glob("node[0-9]*")):
        node_id = node_dir.name.replace("node", "")
        cpulist = read_file(node_dir / "cpulist")
        meminfo = read_file(node_dir / "meminfo")
        mem_total_kb = None
        for line in meminfo.splitlines():
            if line.startswith("MemTotal"):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        mem_total_kb = int(parts[3])
                    except ValueError:
                        pass
        nodes.append(
            {
                "id": int(node_id),
                "cpulist": cpulist,
                "mem_total_kb": mem_total_kb,
            }
        )

    return {"numa_nodes": len(nodes) or 1, "nodes": nodes}


def get_memory_info():
    if psutil:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {
            "total_ram": vm.total,
            "available_ram": vm.available,
            "swap_total": sm.total,
            "swap_used": sm.used,
        }
    else:
        meminfo = read_file("/proc/meminfo")
        kv = parse_key_value_lines(meminfo)
        def kb(key):
            v = kv.get(key)
            if not v:
                return None
            try:
                return int(v.split()[0]) * 1024
            except ValueError:
                return None

        return {
            "total_ram": kb("MemTotal"),
            "available_ram": kb("MemAvailable"),
            "swap_total": kb("SwapTotal"),
            "swap_used": kb("SwapFree"),
        }


def get_storage_info():
    lsblk_out = run_cmd(["lsblk", "-o", "NAME,TYPE,SIZE,ROTA,MOUNTPOINT,FSTYPE", "-J"])
    if lsblk_out:
        try:
            data = json.loads(lsblk_out)
            return data
        except json.JSONDecodeError:
            pass
    return {"blockdevices": []}


def get_gpu_info():
    # Only basic NVIDIA detection; extend as needed
    if not which("nvidia-smi"):
        return {"has_nvidia": False, "gpus": []}

    out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            name, mem, cc = parts[:3]
            try:
                mem_mib = int(mem)
            except ValueError:
                mem_mib = None
            gpus.append(
                {
                    "name": name,
                    "memory_mib": mem_mib,
                    "compute_capability": cc,
                }
            )
    return {"has_nvidia": bool(gpus), "gpus": gpus}


def get_kernel_tuning():
    thp = read_file("/sys/kernel/mm/transparent_hugepage/enabled")
    swappiness = read_file("/proc/sys/vm/swappiness")
    governor = ""
    cpufreq_dir = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    if cpufreq_dir.exists():
        governor = read_file(cpufreq_dir / "scaling_governor")

    irqbalance_running = False
    if psutil:
        for p in psutil.process_iter(attrs=["name"]):
            if p.info.get("name") == "irqbalance":
                irqbalance_running = True
                break
    else:
        ps_out = run_cmd(["ps", "aux"])
        irqbalance_running = "irqbalance" in ps_out

    return {
        "thp_enabled_raw": thp,
        "swappiness": swappiness,
        "governor": governor,
        "irqbalance_running": irqbalance_running,
    }


# -----------------------------
# Benchmarks (lightweight)
# -----------------------------

def benchmark_cpu_python(duration_sec=2.0):
    """
    Very rough CPU benchmark using pure Python.
    Not representative of LLM workloads, but gives a relative feel.
    """
    start = time.time()
    n = 0
    while time.time() - start < duration_sec:
        n += 1
    ops_per_sec = n / duration_sec
    return {"python_loop_ops_per_sec": ops_per_sec}


def benchmark_sysbench_cpu():
    if not which("sysbench"):
        return None
    out = run_cmd(
        ["sysbench", "cpu", "--cpu-max-prime=20000", "run"],
        timeout=30,
    )
    if not out:
        return None
    total_time = None
    events_per_sec = None
    for line in out.splitlines():
        if "total time:" in line:
            total_time = float(line.split()[-2])
        if "events per second:" in line:
            events_per_sec = float(line.split()[-1])
    return {
        "sysbench_total_time_sec": total_time,
        "sysbench_events_per_sec": events_per_sec,
    }


def benchmark_mbw():
    if not which("mbw"):
        return None
    out = run_cmd(["mbw", "-n", "3", "512"], timeout=60)
    if not out:
        return None
    # Parse last line with MB/s
    bw = None
    for line in out.splitlines()[::-1]:
        if "AVG" in line or "MB/s" in line:
            m = re.search(r"([0-9.]+)\s+MB/s", line)
            if m:
                bw = float(m.group(1))
                break
    return {"mbw_bandwidth_mb_s": bw}


# -----------------------------
# Capability & suitability analysis
# -----------------------------

def has_flag(flags, name):
    return name in flags


def analyze_cpu_capabilities(cpu_info):
    flags = cpu_info.get("flags", [])
    return {
        "avx2": has_flag(flags, "avx2"),
        "avx512": any(f.startswith("avx512") for f in flags),
        "fma": has_flag(flags, "fma"),
        "bmi1": has_flag(flags, "bmi1"),
        "bmi2": has_flag(flags, "bmi2"),
        "vnni": any("vnni" in f for f in flags),
        "bf16": any("bf16" in f for f in flags),
    }


def estimate_cpu_llm_capacity(mem_info, cpu_info, gpu_info):
    """
    Very rough, conservative estimates for CPU-only inference.
    Assumes quantized GGUF models and moderate context length.
    """
    total_ram = mem_info.get("total_ram") or 0
    total_ram_gib = total_ram / (1024**3)

    # Simple thresholds (you can tune these)
    # These assume Q4-ish quantization and some headroom for OS.
    capacity = {
        "7B": total_ram_gib >= 16,
        "13B": total_ram_gib >= 32,
        "33B": total_ram_gib >= 64,
        "70B": total_ram_gib >= 128,
    }

    notes = []
    if total_ram_gib < 16:
        notes.append("System RAM is tight for CPU-only LLMs; prefer small 3B–7B models or GPU offload.")
    if cpu_info.get("logical_cores", 0) < 8:
        notes.append("Few logical cores; throughput for concurrent requests will be limited.")
    if not analyze_cpu_capabilities(cpu_info).get("avx2"):
        notes.append("No AVX2 detected; many modern LLM kernels will be significantly slower.")

    # GPU hint
    if gpu_info.get("has_nvidia"):
        notes.append("NVIDIA GPU detected; consider GPU-first inference (TensorRT-LLM, vLLM, etc.).")

    return {"capacity": capacity, "notes": notes}


def interpret_thp(thp_raw):
    """
    /sys/kernel/mm/transparent_hugepage/enabled example:
    [always] madvise never
    """
    if not thp_raw:
        return {"mode": "unknown", "recommended": "madvise"}
    modes = thp_raw.replace("[", "").replace("]", "").split()
    current = None
    for part in thp_raw.split():
        if part.startswith("[") and part.endswith("]"):
            current = part.strip("[]")
            break
    return {"mode": current or "unknown", "available_modes": modes, "recommended": "madvise"}


def interpret_swappiness(swappiness_raw):
    try:
        val = int(swappiness_raw.strip())
    except Exception:
        val = None
    rec = 10
    return {"current": val, "recommended": rec}


def interpret_governor(governor_raw):
    gov = governor_raw.strip() if governor_raw else ""
    rec = "performance"
    return {"current": gov or "unknown", "recommended": rec}


# -----------------------------
# Tuning script generation
# -----------------------------

def generate_tuning_script(kernel_tuning, numa_info, cpu_info):
    thp = interpret_thp(kernel_tuning.get("thp_enabled_raw", ""))
    swap = interpret_swappiness(kernel_tuning.get("swappiness", "60"))
    gov = interpret_governor(kernel_tuning.get("governor", ""))

    lines = [
        "#!/usr/bin/env bash",
        "# Conservative tuning script generated by ai-server-diag.py",
        "# Review before applying. Run as root.",
        "set -e",
        "",
        "echo 'Applying conservative AI server tunings...'",
        "",
        "# Transparent Huge Pages: prefer madvise over always",
        "if [ -d /sys/kernel/mm/transparent_hugepage ]; then",
        "  echo madvise > /sys/kernel/mm/transparent_hugepage/enabled || true",
        "  echo madvise > /sys/kernel/mm/transparent_hugepage/defrag || true",
        "fi",
        "",
        "# Swappiness",
        f"echo {swap['recommended']} > /proc/sys/vm/swappiness || true",
        "sysctl -w vm.swappiness={}".format(swap["recommended"]),
        "",
        "# CPU governor",
        "for cpu in /sys/devices/system/cpu/cpu[0-9]*; do",
        "  if [ -f \"$cpu/cpufreq/scaling_governor\" ]; then",
        f"    echo {gov['recommended']} > \"$cpu/cpufreq/scaling_governor\" || true",
        "  fi",
        "done",
        "",
        "# IRQ balance: usually keep enabled on multi-core systems",
        "# systemctl enable --now irqbalance || true",
        "",
        "# NUMA hint: pin AI workloads to a single NUMA node for better locality.",
        "# Example (adjust node and CPUs):",
        "# numactl --cpunodebind=0 --membind=0 your_llm_server_command",
        "",
        "echo 'Done. Some changes may require reboot or manual GRUB edits for full effect.'",
        "",
    ]

    return "\n".join(lines)


# -----------------------------
# Markdown report generation
# -----------------------------

def generate_markdown_report(
    sys_info,
    cpu_info,
    numa_info,
    mem_info,
    storage_info,
    gpu_info,
    kernel_tuning,
    cpu_caps,
    cpu_bench_py,
    cpu_bench_sysbench,
    mbw_res,
    llm_capacity,
):
    lines = []
    lines.append(f"# AI Server Diagnostics Report")
    lines.append("")
    lines.append(f"Generated: `{datetime.utcnow().isoformat()}Z`")
    lines.append("")

    # System
    lines.append("## System")
    lines.append(f"- **Hostname:** `{sys_info['hostname']}`")
    lines.append(f"- **OS:** `{sys_info['os']}`")
    lines.append(f"- **Kernel:** `{sys_info['kernel']}`")
    lines.append(f"- **Python:** `{sys_info['python']}`")
    lines.append("")

    # CPU
    lines.append("## CPU")
    lines.append(f"- **Model:** `{cpu_info.get('model_name')}`")
    lines.append(f"- **Vendor:** `{cpu_info.get('vendor')}`")
    lines.append(f"- **Physical cores (est.):** `{cpu_info.get('physical_cores')}`")
    lines.append(f"- **Logical cores:** `{cpu_info.get('logical_cores')}`")
    lines.append(f"- **Max freq (MHz):** `{cpu_info.get('max_freq_mhz')}`")
    lines.append(f"- **L1d cache:** `{cpu_info.get('l1d_kb')}`")
    lines.append(f"- **L1i cache:** `{cpu_info.get('l1i_kb')}`")
    lines.append(f"- **L2 cache:** `{cpu_info.get('l2_kb')}`")
    lines.append(f"- **L3 cache:** `{cpu_info.get('l3_kb')}`")
    lines.append("")
    lines.append("### CPU instruction set capabilities")
    lines.append(f"- **AVX2:** `{cpu_caps['avx2']}`")
    lines.append(f"- **AVX-512 (any):** `{cpu_caps['avx512']}`")
    lines.append(f"- **FMA:** `{cpu_caps['fma']}`")
    lines.append(f"- **BMI1/BMI2:** `{cpu_caps['bmi1']}/{cpu_caps['bmi2']}`")
    lines.append(f"- **VNNI (int8-friendly):** `{cpu_caps['vnni']}`")
    lines.append(f"- **BF16:** `{cpu_caps['bf16']}`")
    lines.append("")

    # NUMA
    lines.append("## NUMA topology")
    lines.append(f"- **NUMA nodes:** `{numa_info['numa_nodes']}`")
    for node in numa_info["nodes"]:
        lines.append(
            f"  - Node {node['id']}: CPUs `{node['cpulist']}`, Mem `{human_bytes((node['mem_total_kb'] or 0)*1024)}`"
        )
    lines.append("")

    # Memory
    lines.append("## Memory")
    lines.append(f"- **Total RAM:** `{human_bytes(mem_info['total_ram'])}`")
    lines.append(f"- **Available RAM:** `{human_bytes(mem_info['available_ram'])}`")
    lines.append(f"- **Swap total:** `{human_bytes(mem_info['swap_total'])}`")
    lines.append(f"- **Swap used:** `{human_bytes(mem_info['swap_used'])}`")
    lines.append("")

    # Storage
    lines.append("## Storage")
    if storage_info.get("blockdevices"):
        for dev in storage_info["blockdevices"]:
            name = dev.get("name")
            size = dev.get("size")
            rota = dev.get("rota")
            dtype = dev.get("type")
            fstype = dev.get("fstype")
            mountpoint = dev.get("mountpoint")
            lines.append(
                f"- `{name}` ({dtype}), size `{size}`, rotational `{rota}`, fs `{fstype}`, mount `{mountpoint}`"
            )
    else:
        lines.append("- No storage info available (lsblk JSON failed).")
    lines.append("")

    # GPU
    lines.append("## GPU")
    if gpu_info["has_nvidia"]:
        for i, g in enumerate(gpu_info["gpus"]):
            lines.append(
                f"- GPU {i}: `{g['name']}`, {g['memory_mib']} MiB, CC `{g['compute_capability']}`"
            )
    else:
        lines.append("- No NVIDIA GPU detected or nvidia-smi not available.")
    lines.append("")

    # Kernel tuning
    lines.append("## Kernel & power tuning")
    thp = interpret_thp(kernel_tuning.get("thp_enabled_raw", ""))
    swap = interpret_swappiness(kernel_tuning.get("swappiness", "60"))
    gov = interpret_governor(kernel_tuning.get("governor", ""))
    lines.append(f"- **Transparent Huge Pages:** raw `{kernel_tuning['thp_enabled_raw']}` → mode `{thp['mode']}`, recommended `{thp['recommended']}`")
    lines.append(f"- **Swappiness:** current `{swap['current']}`, recommended `{swap['recommended']}`")
    lines.append(f"- **CPU governor (cpu0):** current `{gov['current']}`, recommended `{gov['recommended']}`")
    lines.append(f"- **irqbalance running:** `{kernel_tuning['irqbalance_running']}`")
    lines.append("")

    # Benchmarks
    lines.append("## Benchmarks (lightweight)")
    lines.append("### Python CPU loop")
    lines.append(f"- **Ops/sec (approx):** `{cpu_bench_py['python_loop_ops_per_sec']:.0f}`")
    lines.append("")
    lines.append("### sysbench cpu (if available)")
    if cpu_bench_sysbench:
        lines.append(f"- **Total time (s):** `{cpu_bench_sysbench['sysbench_total_time_sec']}`")
        lines.append(f"- **Events/sec:** `{cpu_bench_sysbench['sysbench_events_per_sec']}`")
    else:
        lines.append("- sysbench not available or failed.")
    lines.append("")
    lines.append("### mbw memory bandwidth (if available)")
    if mbw_res:
        lines.append(f"- **Bandwidth (MB/s):** `{mbw_res['mbw_bandwidth_mb_s']}`")
    else:
        lines.append("- mbw not available or failed.")
    lines.append("")

    # LLM capacity
    lines.append("## Estimated CPU LLM capacity (very rough)")
    cap = llm_capacity["capacity"]
    lines.append("| Model | Likely feasible on CPU-only? |")
    lines.append("|-------|------------------------------|")
    for model, ok in cap.items():
        lines.append(f"| {model} | {'✅' if ok else '⚠️'} |")
    lines.append("")
    if llm_capacity["notes"]:
        lines.append("### Notes")
        for n in llm_capacity["notes"]:
            lines.append(f"- {n}")
        lines.append("")

    # Recommendations summary
    lines.append("## High-impact recommendations (summary)")
    lines.append("- Set CPU governor to `performance` for dedicated inference nodes.")
    lines.append("- Set Transparent Huge Pages to `madvise` instead of `always`.")
    lines.append("- Reduce `vm.swappiness` to around 10 on inference nodes.")
    lines.append("- Pin LLM workloads to a single NUMA node for better locality.")
    lines.append("- Use a real LLM benchmark (e.g., llama.cpp tokens/sec) for final capacity validation.")
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------

def main():
    sys_info = get_basic_system_info()
    cpu_info = get_cpu_info()
    numa_info = get_numa_info()
    mem_info = get_memory_info()
    storage_info = get_storage_info()
    gpu_info = get_gpu_info()
    kernel_tuning = get_kernel_tuning()
    cpu_caps = analyze_cpu_capabilities(cpu_info)

    cpu_bench_py = benchmark_cpu_python()
    cpu_bench_sysbench = benchmark_sysbench_cpu()
    mbw_res = benchmark_mbw()

    llm_capacity = estimate_cpu_llm_capacity(mem_info, cpu_info, gpu_info)

    report_md = generate_markdown_report(
        sys_info,
        cpu_info,
        numa_info,
        mem_info,
        storage_info,
        gpu_info,
        kernel_tuning,
        cpu_caps,
        cpu_bench_py,
        cpu_bench_sysbench,
        mbw_res,
        llm_capacity,
    )

    tuning_script = generate_tuning_script(kernel_tuning, numa_info, cpu_info)

    out_dir = Path.cwd()
    report_path = out_dir / "ai-server-report.md"
    tuning_path = out_dir / "ai-server-tune.sh"

    report_path.write_text(report_md)
    tuning_path.write_text(tuning_script)
    tuning_path.chmod(0o755)

    print(f"Report written to: {report_path}")
    print(f"Tuning script written to: {tuning_path}")
    print("")
    print("Next steps:")
    print("- Review ai-server-report.md for detailed diagnostics.")
    print("- Carefully inspect ai-server-tune.sh before running it as root.")
    print("- For real LLM performance, benchmark your actual inference stack (e.g., llama.cpp, vLLM).")


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Warning: running as non-root; some information may be incomplete.", file=sys.stderr)
    main()
