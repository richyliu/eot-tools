#!/bin/bash

# check sockets and qemu processes before and after and compare

num_sockets_before=$(find ./tmp_sockets -name "secure_protocol_sockets_*" | wc -l)
num_qemu_before=$(ps aux | grep '[q]emu' | wc -l)

timeout 1 bash -c "./run_qemu.sh <<< 1"
sleep 1

timeout 2 bash -c "./test_orchestrator.py --arm full_pairing"
sleep 1

num_sockets_after=$(find ./tmp_sockets -name "secure_protocol_sockets_*" | wc -l)
num_qemu_after=$(ps aux | grep '[q]emu' | wc -l)

echo "Sockets before: $num_sockets_before"
echo "Sockets after: $num_sockets_after"
echo "QEMU before: $num_qemu_before"
echo "QEMU after: $num_qemu_after"

if [ "$num_sockets_before" -ne "$num_sockets_after" ]; then
    echo "FAIL: Sockets count changed!"
else
    echo "PASS: Sockets count is the same."
fi

if [ "$num_qemu_before" -ne "$num_qemu_after" ]; then
    echo "FAIL: QEMU count changed!"
else
    echo "PASS: QEMU count is the same."
fi
