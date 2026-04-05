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

    print(f"Building ARM binaries for mode: {args.mode}...")
    run_cmd("make clean && make arm")
    print("Building QEMU inscount plugin...")
    run_cmd("make -C qemu_tcg_plugins clean && make -C qemu_tcg_plugins")
    collect_size()
    
    print(f"\n--- Running Orchestrator Tests (ARM/QEMU) [{args.mode}] ---")
    print(f"Simulating real-time latency with baud rate: {args.baud}")
    print("This may take 30-90 seconds...")
    
    # Run tests. Use --parallel for speed, but note it can be sensitive to QEMU scheduling.
    cmd = f"./test_orchestrator.py --arm {args.mode} --baud {args.baud} --parallel"
    run_cmd(cmd)

    def get_latest_log_dir(prefix):
        dirs = glob.glob(f"test_logs/{prefix}_*")
        if not dirs: return None
        return max(dirs, key=os.path.getctime)

    latest_basic_dir = get_latest_log_dir("test_basic_communication")
    latest_legacy_dir = get_latest_log_dir("test_legacy_baseline")
    
    if not latest_basic_dir and not latest_legacy_dir:
        print("No test logs found.")
        return

    def parse_metrics_from_dir(log_dir):
        if not log_dir: return None
        metrics = {
            "cycles": {}, 
            "stack": {"pairing": 0, "status_request": 0, "emergency_brake": 0}, 
            "latency": {"pairing": 0, "status_request": 0, "emergency_brake": 0}, 
            "bandwidth": {
                "total": 0, 
                "pairing": 0,
                "status_request": 0,
                "emergency_brake": 0,
                "other": 0
            }
        }
        eot_log = os.path.join(log_dir, "eot.log")
        hot_log = os.path.join(log_dir, "hot.log")
        
        for log_file in [eot_log, hot_log]:
            if not os.path.exists(log_file): continue
            is_eot = "eot.log" in log_file
            local_max_stack = 0
            
            with open(log_file, "r") as f:
                for line in f:
                    # 1. Cycle/Instruction profiling (aggregated)
                    m_prof = re.search(r"\[PROFILE\] ([\w_]+): (\d+) (cycles|instructions)", line)
                    if m_prof:
                        metrics["cycles"][m_prof.group(1)] = (int(m_prof.group(2)), m_prof.group(3))
                    
                    # 2. Stack Tracking (Monotonic per device)
                    m_stack = re.search(r"Stack: (\d+)/", line)
                    if m_stack:
                        local_max_stack = max(local_max_stack, int(m_stack.group(1)))

                    # 3. Latency & Scenario stack association (Latch at phase completion)
                    if "pairing took" in line:
                        m = re.search(r"(\d+) ms", line)
                        if m: metrics["latency"]["pairing"] = int(m.group(1))
                        metrics["stack"]["pairing"] = max(metrics["stack"]["pairing"], local_max_stack)
                    
                    if "status update took" in line:
                        m = re.search(r"(\d+) ms", line)
                        if m: metrics["latency"]["status_request"] = int(m.group(1))
                        metrics["stack"]["status_request"] = max(metrics["stack"]["status_request"], local_max_stack)
                    
                    if any(x in line for x in ["emergency brake confirmation", "emergency brake acknowledgment"]):
                        m = re.search(r"(\d+) ms elapsed", line)
                        if m: metrics["latency"]["emergency_brake"] = int(m.group(1))
                        metrics["stack"]["emergency_brake"] = max(metrics["stack"]["emergency_brake"], local_max_stack)
                    
                    # 4. Bandwidth (By message type)
                    m_msg = re.search(r"sending message of length (\d+).*msg_type=(\d+)", line)
                    if m_msg:
                        size = int(m_msg.group(1))
                        mtype = int(m_msg.group(2))
                        metrics["bandwidth"]["total"] += size
                        if is_eot:
                            if mtype == 0 or mtype == 1: # PUBKEY, NONCE
                                metrics["bandwidth"]["pairing"] += size
                            elif mtype == 2: # STATUS
                                metrics["bandwidth"]["status_request"] += size
                            elif mtype == 3: # EMERGENCY
                                metrics["bandwidth"]["emergency_brake"] += size
                            else:
                                metrics["bandwidth"]["other"] += size
                        else: # HOT
                            if mtype == 0 or mtype == 1 or mtype == 2: # ADV, PUBKEY_AND_COMMIT, NONCE
                                metrics["bandwidth"]["pairing"] += size
                            elif mtype == 3: # STATUS
                                metrics["bandwidth"]["status_request"] += size
                            elif mtype == 4: # EMERGENCY
                                metrics["bandwidth"]["emergency_brake"] += size
                            else:
                                metrics["bandwidth"]["other"] += size

                    m_legacy = re.search(r"sent legacy message of length (\d+)(?: \(payload=(\d+)\))?", line)
                    if m_legacy:
                        size = int(m_legacy.group(1))
                        metrics["bandwidth"]["total"] += size
                        payload_len = int(m_legacy.group(2)) if m_legacy.group(2) else None
                        if payload_len is not None:
                            if payload_len == 3: # "ARM"
                                metrics["bandwidth"]["pairing"] += size
                            elif payload_len == 4 or payload_len >= 12: # "STAT" or status struct
                                metrics["bandwidth"]["status_request"] += size
                            elif payload_len == 2 or payload_len == 6: # "EB" or "ACK EB"
                                metrics["bandwidth"]["emergency_brake"] += size
                            else:
                                metrics["bandwidth"]["other"] += size
                        else:
                            if size == 11:
                                metrics["bandwidth"]["pairing"] += size
                            elif size == 12 or size == 20:
                                metrics["bandwidth"]["status_request"] += size
                            elif size == 10 or size == 14:
                                metrics["bandwidth"]["emergency_brake"] += size
                            else:
                                metrics["bandwidth"]["other"] += size

        # Fallback latches: Ensure monotonic scenarios have at least current peak if phase completed but marker missed
        metrics["stack"]["status_request"] = max(metrics["stack"]["status_request"], metrics["stack"]["pairing"])
        metrics["stack"]["emergency_brake"] = max(metrics["stack"]["emergency_brake"], metrics["stack"]["status_request"])
        
        return metrics

    basic_metrics = parse_metrics_from_dir(latest_basic_dir)
    legacy_metrics = parse_metrics_from_dir(latest_legacy_dir)

    def print_comparison_table(modern, legacy):
        def get_change(m, l):
            if l == 0:
                return "N/A"
            change = ((m - l) / l) * 100
            return f"{change:>+8.1f}%"

        header = f" {'Metric':<18} | {'Scenario':<18} | {'Legacy':>14} | {'Modern':>14} | {'Change (%)':>12}"
        width = len(header) + 1
        print("\n" + "="*width)
        print(header)
        print("-" * width)
        
        scenarios = [
            ("pairing", "Pairing"),
            ("status_request", "Status Update"),
            ("emergency_brake", "Emergency Brake")
        ]

        # 1. Bandwidth (in bits)
        for s_key, s_name in scenarios:
            m_val = modern["bandwidth"].get(s_key, 0) * 8
            l_val = legacy["bandwidth"].get(s_key, 0) * 8 if legacy else 0
            change = get_change(m_val, l_val)
            print(f" {'Bandwidth':<18} | {s_name:<18} | [{l_val:>4}] , [{m_val:>4}] , [{change:>6}],")
        
        print("-" * width)
        # 2. Latency
        for s_key, s_name in scenarios:
            m_val = modern["latency"].get(s_key, 0)
            l_val = legacy["latency"].get(s_key, 0) if legacy else 0
            change = get_change(m_val, l_val)
            print(f" {'Latency':<18} | {s_name:<18} | [{l_val:>4}] , [{m_val:>4}] , [{change:>6}],")

        print("-" * width)
        # 3. Stack Usage
        for s_key, s_name in scenarios:
            m_val = modern["stack"].get(s_key, 0)
            l_val = legacy["stack"].get(s_key, 0) if legacy else 0
            change = get_change(m_val, l_val)
            print(f" {'Max Stack':<18} | {s_name:<18} | [{l_val:>4}] , [{m_val:>4}] , [{change:>6}],")
        
        print("="*width + "\n")

    if basic_metrics:
        print_comparison_table(basic_metrics, legacy_metrics)
    else:
        print("No modern metrics collected (refer to basic_communication test logs).")

    if basic_metrics and basic_metrics["cycles"]:
        print("--- Detailed Metadata (Modern - Instructions) ---")
        for k, v in sorted(basic_metrics["cycles"].items()):
            print(f"  {k}: {v[0]} {v[1]}")
    
    print("\n--- 5. Packet-Loss Recovery ---")
    print("  TODO: Implement packet loss recovery testing.")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
