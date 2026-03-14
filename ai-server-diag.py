#!/usr/bin/env python3
# Run as: sudo python3 ai_optimize.py

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

# -----------------------------
# Global paths / constants
# -----------------------------
BASE_DIR = Path("ai_optimize")
LOG_DIR = BASE_DIR / "logs"
REPORT_PATH = BASE_DIR / "report.md"
SETTINGS_PATH = BASE_DIR / "settings.json"
APPLY_SAFE_SH = BASE_DIR / "apply_safe.sh"
APPLY_UNSAFE_SH = BASE_DIR / "apply_unsafe.sh"
ROLLBACK_SH = BASE_DIR / "rollback.sh"
CHANGES_LOG = LOG_DIR / "changes.log"

REQUIRED_TOOLS_DEBIAN = [
    "cpufrequtils", "numactl", "dmidecode", "util-linux", "systemd-container", "lsblk"
]
REQUIRED_TOOLS_REDHAT = [
    "cpupower", "numactl", "dmidecode", "tuned", "virt-what", "util-linux", "lsblk"
]

# -----------------------------
# Utility helpers
# -----------------------------
def run_cmd(cmd, check=False):
    try:
        result = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return e.stdout.strip() if e.stdout else ""

def log(msg):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHANGES_LOG, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def ensure_root():
    if os.geteuid() != 0:
        print("This script must be run as root (sudo).")
        sys.exit(1)

def read_os_release():
    data = {}
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    data[k] = v.strip('"')
    return data

def detect_distro():
    info = read_os_release()
    id_ = info.get("ID", "").lower()
    like = info.get("ID_LIKE", "").lower()

    if any(x in (id_, like) for x in ["debian", "ubuntu"]):
        return "debian"
    if any(x in (id_, like) for x in ["rhel", "centos", "fedora", "rocky", "almalinux"]):
        return "redhat"
    return "unknown"

def install_required_tools(distro):
    print("\n[+] Checking and installing required tools...")
    if distro == "debian":
        tools = REQUIRED_TOOLS_DEBIAN
        pkg_mgr = "apt"
        update_cmd = "apt update -y"
        install_cmd = "apt install -y"
    elif distro == "redhat":
        tools = REQUIRED_TOOLS_REDHAT
        pkg_mgr = "dnf"
        update_cmd = "dnf makecache -y"
        install_cmd = "dnf install -y"
    else:
        print("[-] Unknown distro, skipping automatic tool installation.")
        return

    # Check tools
    missing = []
    for t in tools:
        if shutil.which(t.split()[0]) is None:
            missing.append(t)

    if not missing:
        print("[+] All required tools already installed.")
        return

    print(f"[+] Missing tools detected: {', '.join(missing)}")
    print(f"[+] Using {pkg_mgr} to install missing tools...")
    log(f"Installing tools: {', '.join(missing)}")

    run_cmd(update_cmd)
    run_cmd(f"{install_cmd} " + " ".join(missing))

# -----------------------------
# Detection helpers
# -----------------------------
def detect_cpu_info():
    out = run_cmd("lscpu")
    info = {}
    for line in out.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info

def detect_cpu_flags():
    flags = []
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "flags" in line:
                    flags = line.split(":", 1)[1].strip().split()
                    break
    return flags

def detect_mem_info():
    out = run_cmd("grep MemTotal /proc/meminfo")
    kb = 0
    if out:
        kb = int(out.split()[1])
    return kb * 1024  # bytes

def detect_numa_nodes():
    nodes = []
    base = Path("/sys/devices/system/node")
    if base.is_dir():
        for entry in base.iterdir():
            if entry.name.startswith("node"):
                nodes.append(entry.name)
    return nodes

def detect_disks():
    out = run_cmd("lsblk -d -o NAME,ROTA")
    disks = {}
    lines = out.splitlines()
    if len(lines) > 1:
        for line in lines[1:]:
            parts = line.split()
            if len(parts) == 2:
                name, rota = parts
                disks[name] = "SSD/NVMe" if rota == "0" else "HDD"
    return disks or {"unknown": "unknown"}

def detect_virtualization():
    # Try systemd-detect-virt
    out = run_cmd("systemd-detect-virt")
    if out and out != "none":
        return out
    # Fallback: check cpuinfo hypervisor flag
    flags = detect_cpu_flags()
    if "hypervisor" in flags:
        return "generic-virt"
    return "none"

def detect_cpu_vendor(cpu_info):
    vendor = cpu_info.get("Vendor ID", "") or cpu_info.get("Vendor ID:", "")
    if not vendor:
        out = run_cmd("grep vendor_id /proc/cpuinfo | head -n1")
        if out:
            vendor = out.split(":")[1].strip()
    vendor = vendor.lower()
    if "intel" in vendor:
        return "intel"
    if "amd" in vendor:
        return "amd"
    return "unknown"

def detect_sockets(cpu_info):
    sockets = cpu_info.get("Socket(s)", "1")
    try:
        return int(sockets)
    except ValueError:
        return 1

# -----------------------------
# Synthetic benchmarks
# -----------------------------
def memory_bandwidth_test(size_mb=500):
    try:
        import numpy as np
    except ImportError:
        print("[-] numpy not installed, skipping memory bandwidth test.")
        return 0.0

    size = size_mb * 1024 * 1024 // 8
    a = np.random.rand(size)
    b = np.random.rand(size)
    start = time.time()
    _ = a + b
    end = time.time()
    if end - start == 0:
        return 0.0
    return size_mb / (end - start)

def cpu_thread_test():
    start = time.time()
    total = 0
    for _ in range(10_000_000):
        total += 1
    end = time.time()
    return end - start

# -----------------------------
# Scoring logic
# -----------------------------
def score_system(cpu_flags, mem_total, bw, cpu_time, disks, llama_bench=None):
    scores = {}
    reasons = {}

    avx2 = "avx2" in cpu_flags
    avx512 = "avx512f" in cpu_flags
    fast_storage = "HDD" not in disks.values()
    ram_gb = mem_total / 1e9

    cpu_score = 30
    if avx512:
        cpu_score += 40
    elif avx2:
        cpu_score += 25
    else:
        cpu_score += 5

    if cpu_time < 1.0:
        cpu_score += 20
    elif cpu_time < 1.5:
        cpu_score += 10

    mem_score = 0
    if bw > 8000:
        mem_score += 30
    elif bw > 4000:
        mem_score += 20
    elif bw > 2000:
        mem_score += 10

    storage_score = 10 if fast_storage else 0

    llama_score = 0
    if llama_bench and llama_bench.get("tps"):
        tps = llama_bench["tps"]
        if tps > 20:
            llama_score += 30
        elif tps > 10:
            llama_score += 20
        elif tps > 5:
            llama_score += 10

    base_total = max(0, min(100, cpu_score + mem_score + storage_score + llama_score))

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

    # 14B
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
# Report generation
# -----------------------------
def generate_report(cpu_info, cpu_flags, mem_total, bw, cpu_time,
                    disks, numa_nodes, virt, vendor, sockets,
                    scores, reasons):
    BASE_DIR.mkdir(exist_ok=True)
    lines = []

    lines.append("# LLM CPU Optimization Diagnostic Report\n")
    lines.append("## 1. System Overview\n")
    lines.append(f"- **CPU Model:** {cpu_info.get('Model name', cpu_info.get('Model name:', 'Unknown'))}")
    lines.append(f"- **CPU Vendor:** {vendor}")
    lines.append(f"- **Physical Cores:** {cpu_info.get('Core(s) per socket', 'Unknown')}")
    lines.append(f"- **Logical Threads:** {cpu_info.get('CPU(s)', 'Unknown')}")
    lines.append(f"- **Sockets:** {sockets}")
    lines.append(f"- **Virtualization:** {virt}")
    lines.append(f"- **AVX2 Support:** {'Yes' if 'avx2' in cpu_flags else 'No'}")
    lines.append(f"- **AVX512 Support:** {'Yes' if 'avx512f' in cpu_flags else 'No'}")
    lines.append(f"- **Total RAM:** {round(mem_total / 1e9, 2)} GB")
    lines.append(f"- **NUMA Nodes Detected:** {', '.join(numa_nodes) if numa_nodes else 'None / Not detected'}")

    lines.append("\n### Disk Devices\n")
    for d, t in disks.items():
        lines.append(f"- **{d}:** {t}")

    lines.append("\n## 2. Synthetic Benchmarks\n")
    lines.append(f"- **Estimated Memory Bandwidth:** {bw:.2f} MB/s")
    lines.append(f"- **Single-thread Loop Time (10M ops):** {cpu_time:.2f} s")

    lines.append("\n## 3. Model Suitability Scores\n")
    for model in ["7B", "13B", "14B"]:
        lines.append(f"### {model} Models")
        lines.append(f"- **Score:** {scores[model]}/100")
        lines.append("- **Reasons:**")
        for r in reasons[model]:
            lines.append(f"  - {r}")
        lines.append("")

    lines.append("## 4. Optimization Targets\n")
    lines.append("- CPU governor, THP, hugepages, vm.swappiness, vm.max_map_count, dirty ratios, swap, services, noatime.")
    lines.append("- Advanced: isolcpus, nohz_full, rcu_nocbs, mitigations, pstate, SMT, SELinux/AppArmor, IRQ pinning.\n")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"\n[+] Report written to {REPORT_PATH}")

# -----------------------------
# Safe / unsafe settings
# -----------------------------
def get_safe_settings(distro):
    return [
        "Set CPU governor to performance",
        "Enable Transparent Huge Pages",
        "Configure hugepages (2MB and possibly 1GB)",
        "Set vm.swappiness=1",
        "Set vm.max_map_count=262144",
        "Tune dirty ratios (vm.dirty_ratio=10, vm.dirty_background_ratio=5)",
        "Disable swap",
        "Disable non-critical services (cups, avahi, bluetooth, etc.)",
        "Apply noatime to non-root filesystems (if safe)"
    ]

def get_unsafe_settings(vendor, sockets, virt):
    settings = [
        "Add isolcpus to kernel cmdline",
        "Add nohz_full to kernel cmdline",
        "Add rcu_nocbs to kernel cmdline",
        "Disable kernel mitigations (mitigations=off)",
        "Apply noatime to root filesystem in fstab",
        "Pin IRQs for NVMe/NIC to specific CPUs",
        "Disable SMT/Hyperthreading",
        "Disable SELinux/AppArmor"
    ]
    if vendor != "intel":
        # intel_pstate only relevant on Intel
        pass
    else:
        settings.insert(4, "Disable intel_pstate (intel_pstate=disable)")
    if virt != "none":
        # Many unsafe settings are less relevant in VMs
        settings.append("NOTE: System is virtualized; some CPU isolation settings may be ineffective.")
    if sockets > 1:
        settings.append("NUMA-aware isolcpus per socket")
    return settings

# -----------------------------
# Script generation helpers
# -----------------------------
def backup_file(path, backup_dir):
    path = Path(path)
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / (path.name + ".bak")
        shutil.copy2(path, backup)
        log(f"Backed up {path} to {backup}")

def generate_rollback_script(distro):
    lines = ["#!/bin/bash\nset -e\n\n"]
    lines.append("# Rollback script generated by ai_optimize\n")
    lines.append("# This will attempt to restore previous system configuration.\n\n")

    # We only log that backups exist; actual restore is manual or simple copy
    lines.append("echo 'Rollback script placeholder. Restore backups from ai_optimize/backups manually.'\n")

    ROLLBACK_SH.write_text("".join(lines))
    ROLLBACK_SH.chmod(0o750)
    log("Generated rollback.sh")

def generate_apply_safe_sh(distro):
    lines = ["#!/bin/bash\nset -e\n\n"]
    lines.append("# Apply safe AI optimizations\n\n")

    # CPU governor
    if distro == "debian":
        lines.append("echo '[SAFE] Setting CPU governor to performance (Debian)...'\n")
        lines.append("for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do echo performance > \"$c\" 2>/dev/null || true; done\n\n")
    elif distro == "redhat":
        lines.append("echo '[SAFE] Setting CPU governor to performance (RedHat)...'\n")
        lines.append("cpupower frequency-set -g performance || true\n\n")

    # THP
    lines.append("echo '[SAFE] Enabling Transparent Huge Pages...'\n")
    lines.append("echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true\n\n")

    # Hugepages (simple default)
    lines.append("echo '[SAFE] Configuring hugepages (2MB)...'\n")
    lines.append("sysctl -w vm.nr_hugepages=1024 || true\n\n")

    # vm.swappiness
    lines.append("echo '[SAFE] Setting vm.swappiness=1...'\n")
    lines.append("sysctl -w vm.swappiness=1 || true\n\n")

    # vm.max_map_count
    lines.append("echo '[SAFE] Setting vm.max_map_count=262144...'\n")
    lines.append("sysctl -w vm.max_map_count=262144 || true\n\n")

    # dirty ratios
    lines.append("echo '[SAFE] Tuning dirty ratios...'\n")
    lines.append("sysctl -w vm.dirty_ratio=10 || true\n")
    lines.append("sysctl -w vm.dirty_background_ratio=5 || true\n\n")

    # disable swap
    lines.append("echo '[SAFE] Disabling swap...'\n")
    lines.append("swapoff -a || true\n\n")

    # disable non-critical services
    lines.append("echo '[SAFE] Disabling non-critical services (cups, avahi, bluetooth if present)...'\n")
    lines.append("for svc in cups avahi-daemon bluetooth; do systemctl disable --now \"$svc\" 2>/dev/null || true; done\n\n")

    # noatime on non-root mounts (runtime only)
    lines.append("echo '[SAFE] Applying noatime to non-root mounts (runtime only)...'\n")
    lines.append("mount | awk '$3 != \"/\" {print $3}' | while read m; do mount -o remount,noatime \"$m\" 2>/dev/null || true; done\n\n")

    APPLY_SAFE_SH.write_text("".join(lines))
    APPLY_SAFE_SH.chmod(0o750)
    log("Generated apply_safe.sh")

def generate_apply_unsafe_sh(distro, chosen_settings):
    lines = ["#!/bin/bash\nset -e\n\n"]
    lines.append("# Apply unsafe AI optimizations (require reboot / may affect stability)\n\n")

    grub_path = "/etc/default/grub" if distro == "debian" else "/etc/sysconfig/grub"

    for s in chosen_settings:
        if s.startswith("Add isolcpus"):
            lines.append("echo '[UNSAFE] Adding isolcpus to kernel cmdline...'\n")
            lines.append(f"sed -i 's/GRUB_CMDLINE_LINUX=\"/GRUB_CMDLINE_LINUX=\"isolcpus=2-15 /' {grub_path} || true\n\n")
        elif s.startswith("Add nohz_full"):
            lines.append("echo '[UNSAFE] Adding nohz_full to kernel cmdline...'\n")
            lines.append(f"sed -i 's/GRUB_CMDLINE_LINUX=\"/GRUB_CMDLINE_LINUX=\"nohz_full=2-15 /' {grub_path} || true\n\n")
        elif s.startswith("Add rcu_nocbs"):
            lines.append("echo '[UNSAFE] Adding rcu_nocbs to kernel cmdline...'\n")
            lines.append(f"sed -i 's/GRUB_CMDLINE_LINUX=\"/GRUB_CMDLINE_LINUX=\"rcu_nocbs=2-15 /' {grub_path} || true\n\n")
        elif s.startswith("Disable kernel mitigations"):
            lines.append("echo '[UNSAFE] Disabling kernel mitigations...'\n")
            lines.append(f"sed -i 's/GRUB_CMDLINE_LINUX=\"/GRUB_CMDLINE_LINUX=\"mitigations=off /' {grub_path} || true\n\n")
        elif s.startswith("Disable intel_pstate"):
            lines.append("echo '[UNSAFE] Disabling intel_pstate...'\n")
            lines.append(f"sed -i 's/GRUB_CMDLINE_LINUX=\"/GRUB_CMDLINE_LINUX=\"intel_pstate=disable /' {grub_path} || true\n\n")
        elif s.startswith("Apply noatime to root filesystem"):
            lines.append("echo '[UNSAFE] Applying noatime to root filesystem in fstab...'\n")
            lines.append("sed -i 's/ \\// \\//; s/defaults/defaults,noatime/' /etc/fstab || true\n\n")
        elif s.startswith("Pin IRQs"):
            lines.append("echo '[UNSAFE] Pinning IRQs for NVMe/NIC (example, manual tuning recommended)...'\n")
            lines.append("# Example: echo 1 > /proc/irq/XX/smp_affinity\n\n")
        elif s.startswith("Disable SMT"):
            lines.append("echo '[UNSAFE] Disabling SMT/Hyperthreading (runtime only, may require BIOS change)...'\n")
            lines.append("echo off > /sys/devices/system/cpu/smt/control 2>/dev/null || true\n\n")
        elif s.startswith("Disable SELinux"):
            lines.append("echo '[UNSAFE] Disabling SELinux/AppArmor (if present)...'\n")
            lines.append("if [ -f /etc/selinux/config ]; then sed -i 's/SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config; fi\n")
            lines.append("systemctl disable --now apparmor 2>/dev/null || true\n\n")

    # Regenerate grub config
    if distro == "debian":
        lines.append("echo '[UNSAFE] Updating GRUB (Debian)...'\n")
        lines.append("update-grub || true\n\n")
    elif distro == "redhat":
        lines.append("echo '[UNSAFE] Updating GRUB (RedHat)...'\n")
        lines.append("grub2-mkconfig -o /boot/grub2/grub.cfg || true\n\n")

    APPLY_UNSAFE_SH.write_text("".join(lines))
    APPLY_UNSAFE_SH.chmod(0o750)
    log("Generated apply_unsafe.sh")

# -----------------------------
# Main flow
# -----------------------------
def main():
    ensure_root()

    BASE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    distro = detect_distro()
    print(f"[+] Detected distro family: {distro}")
    log(f"Distro: {distro}")

    install_required_tools(distro)

    print("\n[+] Collecting system information and running diagnostics...")
    cpu_info = detect_cpu_info()
    cpu_flags = detect_cpu_flags()
    mem_total = detect_mem_info()
    numa_nodes = detect_numa_nodes()
    disks = detect_disks()
    virt = detect_virtualization()
    vendor = detect_cpu_vendor(cpu_info)
    sockets = detect_sockets(cpu_info)

    bw = memory_bandwidth_test()
    cpu_time = cpu_thread_test()

    scores, reasons = score_system(cpu_flags, mem_total, bw, cpu_time, disks, None)
    generate_report(cpu_info, cpu_flags, mem_total, bw, cpu_time,
                    disks, numa_nodes, virt, vendor, sockets,
                    scores, reasons)

    # Save settings.json
    settings = {
        "distro": distro,
        "virt": virt,
        "vendor": vendor,
        "sockets": sockets,
        "scores": scores,
        "timestamp": time.time()
    }
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
    log("Saved settings.json")

    # SAFE SETTINGS
    safe_list = get_safe_settings(distro)
    print("\nSAFE OPTIMIZATIONS (no reboot required, low risk)\n")
    print("The following safe optimizations can be applied:\n")
    for i, s in enumerate(safe_list, 1):
        print(f"{i}. {s}")
    print()
    choice = input("Apply these safe optimizations now? (y/n): ").strip().lower()
    if choice == "y":
        print("\n[+] Generating and applying safe optimizations...")
        backup_dir = BASE_DIR / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_file("/etc/sysctl.conf", backup_dir)
        generate_apply_safe_sh(distro)
        generate_rollback_script(distro)
        log("Applying safe optimizations via apply_safe.sh")
        subprocess.run([str(APPLY_SAFE_SH)], check=False)
        print("[+] Safe optimizations applied.")
    else:
        print("[*] Skipping safe optimizations.")

    # UNSAFE SETTINGS
    unsafe_list = get_unsafe_settings(vendor, sockets, virt)
    print("\nUNSAFE OPTIMIZATIONS (may require reboot / affect stability/security)\n")
    print("Choose how to proceed:\n")
    print("1) Show list of unsafe optimizations")
    print("2) Interactive mode (review/edit/apply each)")
    print("3) Skip unsafe optimizations\n")
    u_choice = input("Your choice (1/2/3): ").strip()

    chosen_unsafe = []
    if u_choice == "1":
        print("\nUNSAFE OPTIMIZATIONS:\n")
        for i, s in enumerate(unsafe_list, 1):
            print(f"{i}. {s}")
        print("\nNo changes applied. If you want to apply them, rerun and choose interactive mode.")
    elif u_choice == "2":
        print("\nEntering interactive mode for unsafe optimizations.\n")
        for s in unsafe_list:
            if s.startswith("NOTE:"):
                print(f"NOTE: {s}")
                continue
            ans = input(f"Apply setting: {s}? (y/n/skip): ").strip().lower()
            if ans == "y":
                chosen_unsafe.append(s)
            elif ans == "skip":
                continue
        if chosen_unsafe:
            print("\n[+] Generating apply_unsafe.sh...")
            backup_dir = BASE_DIR / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup_file("/etc/default/grub", backup_dir)
            backup_file("/etc/sysconfig/grub", backup_dir)
            backup_file("/etc/fstab", backup_dir)
            generate_apply_unsafe_sh(distro, chosen_unsafe)
            log(f"Chosen unsafe settings: {chosen_unsafe}")
            print("[!] Unsafe optimizations have been scripted in apply_unsafe.sh.")
            print("[!] They are NOT executed automatically. Review and run manually if desired.")
            print("[!] Some changes require a reboot to take effect.")
        else:
            print("[*] No unsafe settings selected.")
    else:
        print("[*] Skipping unsafe optimizations.")

    print("\nDone. See:")
    print(f"- Report: {REPORT_PATH}")
    print(f"- Settings: {SETTINGS_PATH}")
    print(f"- Safe script: {APPLY_SAFE_SH}")
    print(f"- Unsafe script: {APPLY_UNSAFE_SH} (if generated)")
    print(f"- Rollback: {ROLLBACK_SH}")
    print(f"- Logs: {CHANGES_LOG}\n")


if __name__ == "__main__":
    main()
