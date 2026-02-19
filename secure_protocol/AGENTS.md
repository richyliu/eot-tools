# AGENTS.md - Agent Coding Guidelines for secure_protocol

## Project Overview

This is a C project implementing a secure protocol between EOT and HOT devices for train authentication using ECDSA. The project contains:

- **C code**: `main.c`, `devices.c`, `devices.h` - Core protocol implementation
- **Dependencies**: `micro-ecc/` (ECC library), `sha256/` (SHA256 implementation)

## Build Commands

### C Code

Use `make` to build the main executable. Use `./main eot` or `./main hot` to run either the EOT or HOT.

## Testing

To test on Unix machines, use the test_orchestrator.py script. Specifically, run `./test_orchestrator.py full_pairing` to run a single test or `./test_orchestrator.py all` to run all tests (slow, takes up to 60 seconds).

## File Structure

```
secure_protocol/
├── main.c              # Entry point for C implementation
├── devices.c           # C device implementations
├── devices.h           # C header file
├── Makefile            # Build configuration
├── micro-ecc/          # ECC library (third-party)
└── sha256/             # SHA256 library (third-party)
```
