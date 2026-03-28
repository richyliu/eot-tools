#!/usr/bin/env python3
import subprocess
import re
import os
import glob
import argparse

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"Command failed: {cmd}\n{res.stderr}")
    return res.stdout

def collect_size():
    print("\n--- Storage Footprint (ARM) ---")
    out = run_cmd("make arm-size")
    for line in out.splitlines():
        if "text" in line or "eot.elf" in line or "hot.elf" in line:
            print(line)

def main():
    parser = argparse.ArgumentParser(description="Collect benchmarking metrics.")
    parser.add_argument("--mode", choices=["all", "brief"], default="all", help="Test mode (default: all)")
    parser.add_argument("--baud", type=int, default=1200, help="Baud rate for UART simulation (default: 1200, matching hardware)")
    args = parser.parse_args()

    print(f"Building ARM binaries with EVALUATION=1 for mode: {args.mode}...")
    run_cmd("make clean && make arm EVALUATION=1")
    print("Building QEMU inscount plugin...")
    run_cmd("make -C qemu_tcg_plugins clean && make -C qemu_tcg_plugins")
    collect_size()
    
    print(f"\n--- Running Orchestrator Tests (ARM/QEMU) [{args.mode}] ---")
    print(f"Simulating real-time latency with baud rate: {args.baud}")
    print("This may take 30-90 seconds...")
    
    cmd = f"./test_orchestrator.py --arm {args.mode} --baud {args.baud}"
    run_cmd(cmd)

    # Find the newest log directory that contains basic communication (has pairing, status, and EB)
    log_dirs = glob.glob("test_logs/test_basic_communication_*")
    if not log_dirs:
        # Fallback to pairing if basic_communication wasn't run
        log_dirs = glob.glob("test_logs/test_full_pairing_*")
        
    if not log_dirs:
        print("No test logs found.")
        return
    latest_log_dir = max(log_dirs, key=os.path.getctime)
    
    eot_log = os.path.join(latest_log_dir, "eot.log")
    hot_log = os.path.join(latest_log_dir, "hot.log")

    print(f"\nParsing logs from {latest_log_dir} ...\n")
    
    metrics = {"cycles": {}, "stack": [], "latency": {}, "bandwidth": {"total": 0, "payload": 0}}

    for log_file in [eot_log, hot_log]:
        if not os.path.exists(log_file): continue
        with open(log_file, "r") as f:
            for line in f:
                # 1. Cycle/Instruction profiling
                m = re.search(r"\[PROFILE\] ([\w_]+): (\d+) (cycles|instructions)", line)
                if m:
                    metrics["cycles"][m.group(1)] = (int(m.group(2)), m.group(3))
                
                # 2. Stack High-Water Mark profiling
                m = re.search(r"Stack: (\d+)/(\d+) bytes", line)
                if m:
                    metrics["stack"].append((int(m.group(1)), log_file[-7:-4].upper()))

                # 3. Latency tracking
                m = re.search(r"pairing took (\d+) ms", line)
                if m:
                    metrics["latency"]["pairing"] = int(m.group(1))
                m = re.search(r"status update took (\d+) ms", line)
                if m:
                    metrics["latency"]["status_request"] = int(m.group(1))
                m = re.search(r"received emergency brake confirmation from EOT\. (\d+) ms elapsed", line)
                if m:
                    metrics["latency"]["emergency_brake"] = int(m.group(1))
                
                # 4. Bandwidth
                m = re.search(r"sending message of length (\d+) \(payload=(\d+)\)", line)
                if m:
                    metrics["bandwidth"]["total"] += int(m.group(1))
                    metrics["bandwidth"]["payload"] += int(m.group(2))

    print("--- 1. Computational Cost (Cycles/Instructions) ---")
    for k, v in metrics["cycles"].items():
        val, unit = v
        print(f"  {k}: {val} {unit}")

    print("\n--- 2. End-to-End Latency ---")
    for k, v in metrics["latency"].items():
        name = k.replace("_", " ").capitalize()
        print(f"  {name}: {v} ms")

    print("\n--- 3. Peak Memory Usage (Stack High-Water) ---")
    if metrics["stack"]:
        max_eot = max([s[0] for s in metrics["stack"] if s[1] == "EOT"], default=0)
        max_hot = max([s[0] for s in metrics["stack"] if s[1] == "HOT"], default=0)
        print(f"  EOT Peak Stack: {max_eot} bytes")
        print(f"  HOT Peak Stack: {max_hot} bytes")
    else:
        print("  No stack measurements found.")

    print("\n--- 4. Bandwidth Efficiency ---")
    t = metrics["bandwidth"]["total"]
    p = metrics["bandwidth"]["payload"]
    if t > 0:
        print(f"  Total bytes transmitted: {t}")
        print(f"  Payload bytes transmitted: {p}")
        print(f"  Security overhead bytes: {t - p}")
        print(f"  Overhead ratio: {((t - p) / t * 100):.1f}%")

    print("\n--- 5. Packet-Loss Recovery ---")
    print("  Check 'test_packet_drop' output above. If it passed, basic drop recovery is successful.")
    
    print("\nDone.")

if __name__ == "__main__":
    main()

