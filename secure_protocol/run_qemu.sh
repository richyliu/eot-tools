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

# Capture the script PID (which QEMU will inherit via exec)
SCRIPT_PID=$$
(
    # Wait for the transition: Either this process dies OR it gets reparented to 1
    while [ "${SCRIPT_PID:-0}" -gt 0 ] && \
        CURRENT_PPID=$(ps -o ppid= -p "$SCRIPT_PID" 2>/dev/null | tr -d ' ' || echo 0) && \
        [ "${CURRENT_PPID:-0}" -ne 1 ] && \
        kill -0 "$SCRIPT_PID" 2>/dev/null; do
        sleep 1
    done

    # If the parent died (PPID=1), kill QEMU
    FINAL_PPID=$(ps -o ppid= -p "$SCRIPT_PID" 2>/dev/null | tr -d ' ' || echo 0)
    if [ "${FINAL_PPID:-0}" -eq 1 ]; then
        kill -TERM "$SCRIPT_PID" 2>/dev/null || kill -9 "$SCRIPT_PID" 2>/dev/null
    fi

    # Final cleanup of the socket directory
    rm -rf "$SOCKET_DIR"
) &
disown

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
# -icount ...: Sync systick with host, allow WFI to sleep
# -plugin: TCG plugin shared lib
# -d plugin: enable the plugin
exec qemu-system-arm \
    -machine mps2-an386 \
    -cpu cortex-m4 \
    -monitor null \
    -nographic \
    -serial unix:${UART_SOCKET},server,nowait \
    -serial stdio \
    -kernel "$BINARY_NAME" \
    -icount shift=0,align=off,sleep=on \
    -plugin ./qemu/contrib/plugins/libhotblocks.dylib \
    -d plugin \
    $extra_args 2>&1

    # -plugin ./qemu/contrib/plugins/libexeclog.dylib \
    # -plugin ./qemu/contrib/plugins/libhotblocks.dylib \
    # -plugin ./qemu/tests/tcg/plugins/libinsn.dylib \