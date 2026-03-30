#!/bin/bash

set -ex

LOG_FILE="test_all_log.txt"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

make clean
make -C qemu_tcg_plugins clean
make -j$(nproc)
make -C qemu_tcg_plugins -j$(nproc)

echo "Tests started at $(date)"

./test_orchestrator.py all
echo "Unix tests finished at $(date)"

./test_orchestrator.py --arm --baud 1200 all
echo "ARM tests finished at $(date)"

./run_metrics.py
echo "Metrics finished at $(date)"

./test_check_qemu_cleanup.sh
echo "Cleanup check finished at $(date)"

echo "ALL TESTS PASSED AT $(date)"