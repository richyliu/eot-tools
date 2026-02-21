# AGENTS.md - Agent Coding Guidelines for secure_protocol

## Project Overview

This is a C project implementing a secure protocol between EOT and HOT devices for train authentication using ECDSA. The project supports both Unix (for testing) and ARM Cortex-M4 bare metal targets.

## Build Commands

Use `make` to build the main executable.

## Running

Use `./main eot` or `./main hot` to run either the EOT or HOT on Unix.
Use `./run_qemu.sh --eot` to run the EOT on ARM QEMU (similarly for HOT).

Warning: by default, executables will continue running unless killed. Run with a sensible timeout, such as `timeout 5 ./run_qemu.sh`

## Testing

To test on Unix machines, use the test_orchestrator.py script. Specifically, run `./test_orchestrator.py <all>` to run a single test or `./test_orchestrator.py all` to run all tests (slow, takes up to 60 seconds). Run all the tests after a major change or refactor.

## File Structure

```
secure_protocol/
├── main.c                # Entry point (Unix main() and ARM Reset_Handler)
├── devices.c             # Core protocol implementation (EOT/HOT state machines)
├── devices.h             # Header for device functions
├── crypto.c              # Cryptographic operations (ECDSA, hashing)
├── crypto.h              # Header for cryptographic functions
├── comm.h                # Communication abstraction layer header
├── comm_socket.c         # Unix socket implementation of communication
├── comm_arm.c            # ARM bare metal stub for communication
├── arm_support.c         # ARM bare metal support (memset, memcpy)
├── arm_linker.ld         # ARM linker script
├── ext_support.h        # External support abstraction header
├── ext_support.c        # Unix support implementations (I/O, random, timer)
├── ext_support_arm.c    # ARM bare metal support stubs
├── nanoprintf.h         # Third-party tiny printf implementation
├── run_qemu.sh          # QEMU launch script for ARM testing
├── devices.py           # Python reference implementation of protocol
├── test_orchestrator.py # Test orchestrator for protocol testing
├── Makefile             # Build configuration
├── README.md            # Project documentation
├── requirements.txt    # Python dependencies
├── micro-ecc/           # ECC library (third-party)
└── sha256/              # SHA256 library (third-party)
```

## Architecture

The codebase uses abstraction layers to support both Unix (for testing) and ARM bare metal targets:

- **Communication (`comm_*`)**: Unix sockets for testing, stubs for ARM
- **External Support (`ext_support_*`)**: I/O, random, timer, and memory functions
