#!/bin/bash

set -ex

make clean
make -C qemu_tcg_plugins clean
make -j$(nproc)
make -C qemu_tcg_plugins -j$(nproc)

./test_orchestrator.py all
./test_orchestrator.py --arm all

./run_metrics.py

./test_check_qemu_cleanup.sh