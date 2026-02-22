#!/bin/bash

# Script to run QEMU Cortex-M4 system for ARM bare metal binary
# Supports both EOT and HOT devices with:
#   - UART0: forwarded to Unix socket for device-to-device communication
#   - UART1: forwarded to stdio for I/O (test orchestrator interaction)

set -e

# Default binary name
BINARY_NAME="eot.elf"

GDB_MODE=false

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
        -b|--binary)
            BINARY_NAME="$2"
            shift 2
            ;;
        -g|--gdb)
            GDB_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  -eot, --eot     Run EOT device (default)"
            echo "  -hot, --hot     Run HOT device"
            echo "  -b, --binary    Specify binary file"
            echo "  -g, --gdb       Enable GDB debugging (port 1234)"
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

# Create unique socket directory
SOCKET_DIR=$(mktemp -d /tmp/secure_protocol_XXXXXX)

qemu_pid=""

cleanup() {
    if [[ -n "$qemu_pid" ]]; then
        kill -TERM "$qemu_pid" 2>/dev/null || true
        wait "$qemu_pid" 2>/dev/null || true
    fi
    rm -rf "$SOCKET_DIR"
}

trap cleanup EXIT INT TERM HUP

# Determine UART socket name based on device type
if [[ "$BINARY_NAME" == "eot.elf" ]]; then
    UART_SOCKET="${SOCKET_DIR}/eot_uart.sock"
else
    UART_SOCKET="${SOCKET_DIR}/hot_uart.sock"
fi

echo "UART_SOCKET_DIR=$SOCKET_DIR"

extra_args=""
if [ "$GDB_MODE" = true ]; then
    extra_args="-s -S"
    echo "GDB debugging enabled. Connect to localhost:1234 with GDB." >&2
fi

# Run QEMU with Cortex-M4
# -cpu cortex-m4: Use Cortex-M4 CPU
# -machine mps2-an386: Use MPS2 platform (compatible with Cortex-M4)
# -monitor null: Disable QEMU monitor
# -nographic: Disable graphical output, use serial for all I/O
# -serial unix:... : UART0 -> Unix socket for device-to-device communication
# -serial stdio: UART1 -> stdio for I/O (test orchestrator interaction)
# -kernel: Specify the ELF binary
qemu-system-arm \
    -machine mps2-an386 \
    -cpu cortex-m4 \
    -monitor null \
    -nographic \
    -serial unix:${UART_SOCKET},server,nowait \
    -serial stdio \
    -kernel "$BINARY_NAME" \
    $extra_args </dev/stdin >/dev/stdout 2>&1 &

qemu_pid=$!
wait "$qemu_pid"
