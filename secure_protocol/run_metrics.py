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
    
    cmd = f"./test_orchestrator.py --arm {args.mode} --baud {args.baud} --parallel"
    run_cmd(cmd)

    def get_latest_log_dir(prefix):
        dirs = glob.glob(f"test_logs/{prefix}_*")
        return max(dirs, key=os.path.getctime) if dirs else None

    latest_basic_dir = get_latest_log_dir("test_basic_communication")
    latest_legacy_dir = get_latest_log_dir("test_legacy_baseline")
    
    if not latest_basic_dir and not latest_legacy_dir:
        print("No test logs found.")
        return

    def parse_metrics_from_dir(log_dir):
        if not log_dir: return None
        metrics = {
            "cycles": {}, 
            "stack": [], 
            "latency": {}, 
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
                    m = re.search(r"(?:EOT|HOT): pairing took (\d+) ms", line)
                    if m: metrics["latency"]["pairing"] = int(m.group(1))
                    
                    m = re.search(r"(?:EOT|HOT): status update took (\d+) ms", line)
                    if m: metrics["latency"]["status_request"] = int(m.group(1))
                    
                    m = re.search(r"received emergency brake confirmation from EOT\. (\d+) ms elapsed", line)
                    if m: metrics["latency"]["emergency_brake"] = int(m.group(1))
                    
                    m = re.search(r"legacy pairing took (\d+) ms", line)
                    if m: metrics["latency"]["legacy_pairing"] = int(m.group(1))
                    
                    m = re.search(r"legacy status update took (\d+) ms", line)
                    if m: metrics["latency"]["legacy_status_request"] = int(m.group(1))
                    
                    m = re.search(r"Received legacy emergency brake acknowledgment\. (\d+) ms elapsed", line)
                    if m: metrics["latency"]["legacy_emergency_brake"] = int(m.group(1))
                    
                    if "received unsolicited legacy status update" in line:
                        metrics["latency"]["legacy_unsolicited_status"] = "N/A (unsolicited recorded)"
                    
                    # 4. Bandwidth
                    m = re.search(r"sending message of length (\d+).*msg_type=(\d+)", line)
                    if m:
                        size = int(m.group(1))
                        mtype = int(m.group(2))
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

                    m = re.search(r"sent legacy message of length (\d+)(?: \(payload=(\d+)\))?", line)
                    if m:
                        size = int(m.group(1))
                        metrics["bandwidth"]["total"] += size
                        # Scenario heuristic based on payload length
                        payload_len = int(m.group(2)) if m.group(2) else None
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
                            # Fallback to total lengths if payload info is missing
                            if size == 11:
                                metrics["bandwidth"]["pairing"] += size
                            elif size == 12 or size == 20:
                                metrics["bandwidth"]["status_request"] += size
                            elif size == 10 or size == 14:
                                metrics["bandwidth"]["emergency_brake"] += size
                            else:
                                metrics["bandwidth"]["other"] += size
                
        return metrics

    basic_metrics = parse_metrics_from_dir(latest_basic_dir)
    legacy_metrics = parse_metrics_from_dir(latest_legacy_dir)

    def print_metrics(name, m):
        print(f"\n================ Metrics: {name} ================")
        if not m:
            print("  Not available.")
            return
            
        print("--- 1. Computational Cost (Cycles/Instructions) ---")
        for k, v in m["cycles"].items():
            val, unit = v
            print(f"  {k}: {val} {unit}")

        print("\n--- 2. End-to-End Latency ---")
        for k, v in m["latency"].items():
            name = k.replace("_", " ").capitalize()
            unit_str = " ms" if isinstance(v, int) else ""
            print(f"  {name}: {v}{unit_str}")

        print("\n--- 3. Peak Memory Usage (Stack High-Water) ---")
        if m["stack"]:
            max_eot = max([s[0] for s in m["stack"] if s[1] == "EOT"], default=0)
            max_hot = max([s[0] for s in m["stack"] if s[1] == "HOT"], default=0)
            print(f"  EOT Peak Stack: {max_eot} bytes")
            print(f"  HOT Peak Stack: {max_hot} bytes")
        else:
            print("  No stack measurements found.")

        print("\n--- 4. Bandwidth Efficiency ---")
        t = m["bandwidth"]["total"]
        if t > 0:
            print(f"  Total transmitted:   {t} bytes ({t*8} bits)")
            scenarios = ["pairing", "status_request", "emergency_brake"]
            for s in scenarios:
                val = m["bandwidth"][s]
                name = s.replace("_", " ").capitalize()
                print(f"    {name}: {val} bytes ({val*8} bits)")
            if m["bandwidth"]["other"] > 0:
                print(f"    Other:           {m['bandwidth']['other']} bytes ({m['bandwidth']['other']*8} bits)")

    print_metrics("Modern Protocol (basic_communication)", basic_metrics)
    print_metrics("Legacy Baseline", legacy_metrics)

    print("\n--- 5. Packet-Loss Recovery ---")
    print("  TODO: Implement packet loss recovery testing.")
    
    print("\nDone.")

if __name__ == "__main__":
    main()

