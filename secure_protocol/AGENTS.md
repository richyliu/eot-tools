# AGENTS.md - Agent Coding Guidelines for secure_protocol

## Project Overview

This is a C project implementing a secure protocol between EOT and HOT devices for train authentication using ECDSA. The project supports both Unix (for testing) and ARM Cortex-M4 bare metal targets.

## Build Commands

### C Code

Use `make` to build the main executable. Use `./main eot` or `./main hot` to run either the EOT or HOT.

## Testing

To test on Unix machines, use the test_orchestrator.py script. Specifically, run `./test_orchestrator.py full_pairing` to run a single test or `./test_orchestrator.py all` to run all tests (slow, takes up to 60 seconds).

## File Structure

```
secure_protocol/
├── main.c                # Entry point (Unix main() and ARM Reset_Handler)
├── devices.c             # Core protocol implementation (EOT/HOT state machines)
├── devices.h             # Header for device functions
├── comm.h                # Communication abstraction layer header
├── comm_socket.c         # Unix socket implementation of communication
├── comm_arm.c            # ARM bare metal stub for communication
├── arm_support.c         # ARM bare metal support (memset, memcpy)
├── ext_utils.h           # Utility functions abstraction header
├── ext_utils.c           # Unix utility implementations
├── ext_utils_arm.c       # ARM utility implementations
├── ext_io.h              # I/O abstraction layer header
├── ext_io.c              # Unix I/O implementation (stdin/stdout)
├── ext_io_arm.c          # ARM bare metal I/O stub
├── ext_random.h          # Randomness abstraction layer header
├── ext_random.c          # Unix randomness implementation (/dev/urandom)
├── ext_random_arm.c      # ARM bare metal randomness stub
├── ext_timer.h           # Timer abstraction layer header
├── ext_timer.c           # Unix timer implementation (clock_gettime)
├── ext_timer_arm.c       # ARM bare metal timer stub
├── nanoprintf.h          # Third-party tiny printf implementation
├── devices.py            # Python reference implementation of protocol
├── test_orchestrator.py  # Test orchestrator for protocol testing
├── Makefile              # Build configuration
├── micro-ecc/            # ECC library (third-party)
└── sha256/               # SHA256 library (third-party)
```

## Architecture

The codebase uses abstraction layers to support both Unix (for testing) and ARM bare metal targets:

- **Communication (`comm_*`)**: Unix sockets for testing, stubs for ARM
- **I/O (`ext_io_*`)**: stdin/stdout for testing, stubs for ARM
- **Random (`ext_random_*`)**: /dev/urandom for testing, stubs for ARM
- **Timer (`ext_timer_*`)**: clock_gettime for testing, stubs for ARM
- **Utilities (`ext_utils_*`)**: Memory functions (same implementation for both)
