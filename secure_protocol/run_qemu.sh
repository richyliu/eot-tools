#!/bin/bash

# Script to run QEMU Cortex-M4 system with semihosting for ARM bare metal binary
# Supports both EOT and HOT devices

set -e

# Default binary name
BINARY_NAME="eot.elf"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -eot|--eot)
            BINARY_NAME="eot.elf"
            shift
            ;;
        -hot|--hot)
            BINARY_NAME="hot.elf"
            shift
            ;;
        -bin|--binary)
            BINARY_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  -eot, --eot     Run EOT device (default)"
            echo "  -hot, --hot     Run HOT device"
            echo "  -bin, --binary  Specify binary file"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

# Check if binary exists
if [ ! -f "$BINARY_NAME" ]; then
    echo "Error: Binary file '$BINARY_NAME' not found"
    echo "Please build the ARM binary first: make arm"
    exit 1
fi

echo "Running QEMU Cortex-M4 with semihosting for: $BINARY_NAME"
echo "Use Ctrl+A then X to exit QEMU"

# Run QEMU with Cortex-M4 and semihosting
# -cpu cortex-m4: Use Cortex-M4 CPU
# -machine mps2-an386: Use MPS2 platform (compatible with Cortex-M4)
# -nographic: Disable graphical output (use console)
# -monitor null: Disable QEMU monitor
# -semihosting: Enable semihosting for stdio
# -kernel: Specify the ELF binary
# -monitor telnet:127.0.0.1:1234,server,nowait: Alternative monitor access if needed
qemu-system-arm \
    -machine mps2-an386 \
    -cpu cortex-m4 \
    -nographic \
    -monitor null \
    -semihosting \
    -kernel "$BINARY_NAME"

# Alternative machine if MPS2 doesn't work:
# -machine lm3s6965evb (for Cortex-M3 compatibility)
# -machine versatilepb (generic ARM platform)

# For debugging with GDB:
# qemu-system-arm \
#     -cpu cortex-m4 \
#     -machine mps2-an385 \
#     -nographic \
#     -monitor null \
#     -semihosting \
#     -kernel "$BINARY_NAME" \
#     -s -S  # Wait for GDB connection on port 1234
