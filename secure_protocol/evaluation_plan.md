# Secure Protocol Evaluation Plan

This document outlines the detailed measurement plan for evaluating the operational feasibility and runtime cost of the secure end-of-train (EOT) communication protocol. The methodology and metrics detailed below aim to assess the viability of the protocol on constrained embedded hardware, ensuring safety and real-time operational requirements are met.

## 1. Experimental Methodology

To ensure robust and reproducible results, the following experimental guidelines must be strictly adhered to:

### 1.1 Baseline Comparison
Define and implement a baseline **insecure protocol** (i.e., functioning communication without cryptographic features, authentication, or secure session metadata). Every measured metric must be compared against this baseline to isolate the overhead introduced by the security mechanisms.

### 1.2 Statistical Rigor
Run each test scenario a statistically significant number of times (minimum of 100 iterations per scenario). For every measurement, report the **mean**, **minimum**, **maximum**, and **standard deviation**.

### 1.3 Measurement Isolation
Design the test harness to clearly separate the measurement of **computational cost** (cryptographic operations, packet parsing) from **communication cost** (radio transmission delays, bandwidth availability). 

### 1.4 Controlled Environment
Ensure that all tests—both baseline and secure—are executed on the exact same hardware target. Use identical compiler settings (e.g., `-O2` or `-Os`) and optimization levels to eliminate compiler-induced variance.

### 1.5 Detailed Instrumentation Logs
Record the complete test environment configuration for every run. This includes packet payload sizes, timeout and retry parameters, simulated channel loss rates, and any baseline assumptions about the communication medium.

---

## 2. Measurement Checklist & Metrics Rationale

The evaluation is broken down into six principal assessment areas characterizing computational limits, temporal constraints, memory boundaries, and network resilience.

> [!NOTE]  
> Each measurement requires parallel execution on both the EOT and HOT devices under the described statistical rigor.

### 2.1 Computational Cost (Instruction / Cycle Counts)
- [ ] Measure total cycle counts for **Pairing & Key Exchange**.
- [ ] Measure total cycle counts for **Message Generation & Authentication (HMAC generation)**.
- [ ] Measure total cycle counts for **Message Parsing & Verification**.
- [ ] Measure total cycle counts for **Session Teardown**.

**Rationale:**  
Cycle and instruction counts demonstrate whether the complex cryptographic operations (such as ECDH) are practically executable on a low-power embedded microcontroller (e.g., ARM Cortex-M4 without hardware acceleration). The protocol must not stall the MCU to the point of watch-dog failure or disrupt concurrent high-priority tasks.

### 2.2 Temporal Performance (End-to-End Latency)
- [ ] Measure initiation-to-completion time for a full authenticated payload sequence.
- [ ] Break down the latency timeline: Computation Time vs. On-Wire Transmission Time.
- [ ] Record worst-case (maximum) latency observed during heavy cryptographic phases.

**Rationale:**  
EOT/HOT workflows govern safety-critical train functions (e.g., emergency braking). Therefore, communication is heavily time-sensitive. The security framework must not introduce computational or transmission delays that exceed the operational thresholds required for safe train management.

### 2.3 Working Memory Limits (Peak Memory Usage)
- [ ] Profile the stack high-water mark during the most intensive operation (e.g., point multiplication during ECDH).
- [ ] Measure any dynamic memory allocation (heap usage), if employed.
- [ ] Differentiate between transient memory (stack usage during crypto calls) and persistent memory (long-lived session keys and context structs).

**Rationale:**  
The target MCU operates with severe static RAM limits. Cryptographic libraries are historically memory-intensive. Reaching or exceeding the available SRAM due to deep call stacks or large state structs will result in catastrophic faults unsuited for a deployable legacy device.

### 2.4 Storage Constraints (Flash & SRAM Footprint)
- [ ] Capture the compiled `.text` and `.data`/`.bss` sizes for the baseline protocol.
- [ ] Capture the compiled sizes for the secure protocol.
- [ ] Isolate the footprint of the specific components: Protocol Logic, Cryptographic Library (`micro-ecc`, `sha256`), Support Utilities, and Testing Harness.

**Rationale:**  
Firmware updates to legacy train hardware are constrained by fixed and often strictly partitioned flash memory banks. The entire secure implementation must fit within the existing headroom of deployed devices alongside the pre-existing train control application.

### 2.5 Reliability (Packet-Loss Recovery Behavior)
- [ ] Subject the communication channel to controlled packet loss, duplication, and reordering.
- [ ] Measure the protocol's ability to successfully recover and complete session handshakes.
- [ ] Record the number of necessary protocol retries or message retransmissions to reach a successful state under defined error rates (e.g., 5%, 10%, 20% loss).

**Rationale:**  
The physical operating environment of EOT/HOT systems (train yards, mountainous terrain) relies on highly lossy RF links. A secure protocol must remain robust; if a dropped packet fundamentally breaks the state machine or requires a full, expensive re-keying, the communication link will fall below safe operational usability constraints in the field.

### 2.6 Bandwidth Efficiency (Byte-Transfer Overhead)
- [ ] Count total bytes transmitted and received per logical protocol phase.
- [ ] Calculate the byte difference between the secure protocol and the insecure baseline.
- [ ] Quantify the specific overhead introduced by security layers: identifiers, nonces, MACs, padding, and session metadata.

**Rationale:**  
Embedded train radios operate on extremely narrowband channels with low data rates. Significant byte-overhead from adding large authentication tags, public keys, or nonces directly increases radio airtime. This increases the probability of mid-air collision, extends latency, and drains power constraints, impacting the fundamental efficiency of the telemetry link.
