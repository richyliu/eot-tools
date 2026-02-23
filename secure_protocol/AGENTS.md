# AGENTS.md - Agent Coding Guidelines for secure_protocol

## Project Overview

This is a C project implementing a secure protocol between EOT and HOT devices for train authentication using ECDH for key exchange and HMAC-SHA256 for message authentication. The project supports both Unix (for testing) and ARM Cortex-M4 bare metal targets (STM32F4).

## Build Commands

Use `make` or `make unix` to build for Unix.
Use `make TARGET=arm` or `make arm` to build for ARM.

## Running

- **Unix**: Use `./eot` or `./hot` to run either device.
- **ARM (QEMU)**: Use `./run_qemu.sh --eot` or `--hot`.

Warning: ARM targets on QEMU often hang on exit; use `timeout 10 ./run_qemu.sh --eot` for automated runs.

## Testing

To test the protocol flow, use `test_orchestrator.py`:
- **Unix**: `./test_orchestrator.py all`
- **ARM/QEMU**: `./test_orchestrator.py --arm all`

## File Structure

```
secure_protocol/
├── main.c                # Entry point (Unix main() and ARM Reset_Handler)
├── devices.c             # Core protocol implementation (EOT/HOT state machines)
├── devices.h             # Header for device functions
├── crypto.c              # Cryptographic operations (ECDH, HMAC-SHA256)
├── crypto.h              # Header for cryptographic functions
├── comm.h                # Communication abstraction layer header
├── comm_common.c         # Shared protocol logic (packet framing, HMAC)
├── comm_socket.c         # Unix socket implementation of communication
├── comm_arm.c            # ARM bare metal implementation (UART0)
├── arm_support.c         # ARM bare metal support (memset, memcpy)
├── arm_linker.ld         # ARM linker script
├── ext_support.h        # External support abstraction header
├── ext_support.c        # Unix support implementations (I/O, random, timer)
├── ext_support_arm.c    # ARM bare metal support (UART1 for I/O, SysTick)
├── nanoprintf.h         # Third-party tiny printf implementation
├── run_qemu.sh          # QEMU launch script for ARM testing
├── devices.py           # Python reference implementation of protocol
├── test_orchestrator.py # Test orchestrator for protocol testing
├── Makefile             # Build configuration
├── README.md            # Project documentation
├── requirements.txt      # Python dependencies
├── micro-ecc/           # ECC library (third-party)
└── sha256/              # SHA256 library (third-party)
```

## Architecture

The codebase uses abstraction layers to support both Unix (for testing) and ARM bare metal targets:

- **Communication (`comm_*`)**: 
    - Unix: Uses domain sockets.
    - ARM: Uses **UART0** for inter-device messaging.
- **External Support (`ext_support_*`)**: 
    - Unix: Standard C library.
    - ARM: Custom implementations for I/O (via **UART1**), timing (SysTick), and PRNG (xorshift32 seeded by SysTick).
- **Cryptography**: Uses `micro-ecc` for P-256 ECDH and a custom HMAC-SHA256. Note: ECDH is used to establish a shared secret, which is then used with HMAC-SHA256 to authenticate messages.
