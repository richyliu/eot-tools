# SAFE-T Design

## 1. Design Principles

- SAFE-T is governed by five constraints derived from the operational
  and hardware realities of the North American freight rail environment

### 1.1 Backward Compatibility

- SAFE-T must interoperate with unmodified legacy counterparts; all four
  pairing combinations (updated/legacy × EOT/HOT) must resolve
  gracefully
- An updated EOT paired with a legacy HOT falls back to the unmodified
  S-5701 TEST broadcast; an updated HOT paired with a legacy EOT enters
  the standard legacy arming workflow after the operator enters the
  5-digit unit ID [Ref:AARS5701]
- **Downgrade protection**: legacy mode requires the rear worker to hold
  TEST for approximately 5 seconds; no radio message alone can force
  either device into the insecure protocol path, preventing silent
  attacker-induced downgrade [Ref:ICSA]
- **Legacy-device assumption**: the SAFE-T state machine does not treat
  legacy behavior as a cryptographic failure. Instead, a legacy peer is
  modeled as a device that never emits SAFE-T messages and only responds
  to the unmodified TEST/ARM workflow; this is important for backward
  compatibility and for the formal model, which separates secure and
  legacy control paths
- **Single-operator variant**: some field workflows allow one person to
  complete the pairing sequence by using a linker or remote-arming tool.
  SAFE-T treats this as a deployment-mode variation, not a protocol
  primitive; the secure pairing state machine still assumes the same
  authenticated message exchange, while the auxiliary tool may relay the
  out-of-band PIN or physical confirmation step [Ref:VimeoLinkHETETD]
  [Ref:PatentRemoteArming] [Ref:PatentHOTEmergencyCommands]
- **Selective jamming residual risk**: an adversary who jams only
  `HOT_MSG_ADV` while leaving legacy TEST broadcasts unaffected can
  cause two updated devices to each believe the other is legacy,
  inducing downgrade without physical operator action [Ref:ICSA]; this
  is acknowledged as a non-goal within the jamming scope established in
  §4.2 of the Adversary and Security Model chapter; operators should
  treat a failed SAFE-T pairing between two known-updated devices as a
  security-relevant anomaly requiring investigation, not a routine
  fallback
  - Cross-reference: §2.2 of the Adversary and Security Model chapter
    discusses the selective jamming downgrade attack primitive
- /Recommended figure: protocol decision tree showing how each of the
  four updated/legacy pairing combinations resolves, annotated with the
  physical operator actions required to enter legacy mode/

### 1.2 Minimal Computational and Memory Overhead

- Primitive selection is driven by the STM32F103ZET6 Cortex-M3: 72 MHz,
  no hardware FPU, 512 KB flash, 64 KB SRAM [Ref:STM32F103Datasheet]
  [Ref:ICSA]
- **HMAC-SHA-256** for message authentication: no IV management,
  efficiently computable in integer-only ANSI C, security reduces to
  SHA-256 collision resistance under standard assumptions
- **ECDH over Curve25519 / X25519** for key exchange [Ref:Bernstein2006]
  [Ref:RFC7748]: designed for constrained platforms, avoids NIST
  P-curve implementation pitfalls, 128-bit security with 256-bit keys,
  no FPU required; the micro-ecc library [Ref:GithubMicroecc] provides
  an ARM-optimized implementation with no dynamic memory allocation
- The reference implementation currently uses `uECC_secp256r1()` in the
  `micro-ecc` wrapper, so the concrete prototype instantiates the ECDH
  primitive on NIST P-256; the design intent, threat analysis, and table
  below therefore distinguish between the implementation curve and the
  protocol's target curve choice
  - All cryptographic operations compile to approximately 20 KB on
    Cortex-M4; see Implementation chapter for Cortex-M3 measurements
- **Precomputed emergency brake HMAC**: immediately after pairing, the
  EOT computes and stores the expected EB HMAC in the background; the
  on-path receive handler reduces to a single constant-time comparison,
  well within the 1-second actuation deadline of 49 CFR § 232.405
  [Ref:CFR405]

#### 1.2.1 Cryptographic primitive comparison

- Shared secrets are established via peer-to-peer ECDH authenticated by
  an out-of-band human verification step; no certificate authority,
  revocation infrastructure, or internet connectivity is required
  [Ref:BluetoothSpec]
- Directly motivated by dark-territory operations: EOTDs routinely
  operate with no cellular, PTC backhaul, or internet connectivity;
  PKI-dependent solutions (NGHE 802.16t) either fail closed—rendering
  the brake system inoperable—or fail open, accepting unverifiable
  credentials [Ref:Ondas] [Ref:IndustrialCyber]
- The shared secret is local to a single pairing session; compromise of
  one session does not affect any other EOT/HOT pair or future sessions

### 1.4 Incremental Deployability

- SAFE-T is deployable as a pure firmware update with no hardware
  modifications [Ref:ICSA]; all communication uses the existing 450 MHz UHF
  link and 64-bit S-5701 block framing, with higher-layer packet
  segmentation added above the physical layer
- Rollout can proceed device-by-device across the ~70,000 HOT and EOT
  units in the fleet; the backward compatibility mechanism handles all
  mixed-fleet states during the transition [Ref:IndustrialCyber] [Ref:Ondas]

### 1.5 Original EOT/HOT pairing workflow

- The legacy EOT/HOT protocol was built around a simple operator-mediated
  arming ceremony rather than cryptographic authentication. In the patent
  description, the rear worker enters the EOT's unit ID at the locomotive,
  then activates the EOT test/arming switch, which causes the EOT to
  broadcast a request. The engineer then confirms the pairing from the
  cab by pressing ARM within a short confirmation window [Ref:PatentHOTEmergencyCommands]
- That workflow already reveals the main trust assumption of the original
  system: radio traffic is accepted as genuine once it matches the known
  unit ID and the crew completes the prescribed physical confirmation
  steps. The patent frames the exchange as an operational safeguard, not
  as a security boundary, and it does not describe any mutual
  authentication of the radio messages themselves [Ref:PatentHOTEmergencyCommands]
- This matters for SAFE-T because the new protocol keeps the same crew
  actions and timing intuition, but replaces the implicit trust in raw RF
  broadcasts with explicit cryptographic proof. The comparison is useful:
  the original workflow depends on operator discipline and obscurity,
  while SAFE-T adds session-specific key establishment, commitment, and
  HMAC protection without changing the basic human workflow

- SAFE-T is deployable as a pure firmware update with no hardware
  modifications [Ref:ICSA]; all communication uses the existing 450 MHz UHF
  link and 64-bit S-5701 block framing, with higher-layer packet
  segmentation added above the physical layer
- Rollout can proceed device-by-device across the ~70,000 HOT and EOT
  units in the fleet; the backward compatibility mechanism handles all
  mixed-fleet states during the transition [Ref:IndustrialCyber] [Ref:Ondas]

### 1.6 Commitment Scheme for Known-PIN Attack Prevention

- Without a commitment scheme, either party could observe the other's
  nonce before committing to its own, biasing the derived PIN toward a
  known value [Ref:BluetoothSpec]
- Protocol: HOT generates nonce Nb, computes commitment C = SHA-256(Nb),
  and sends C to the EOT **before** the EOT transmits its nonce Na; the
  EOT sends Na; the HOT then reveals Nb; the EOT verifies SHA-256(Nb) ==
  C before proceeding
- Because the HOT commits to Nb before learning Na, neither party can
  bias PIN = g(Pa, Pb, Na, Nb); the PIN is effectively random from the
  perspective of any party not controlling both nonces
- Directly analogous to the commitment function in Bluetooth Secure
  Simple Pairing Numeric Comparison [Ref:BluetoothSpec]

## 2. Protocol Overview

### 2.1 Packet Format

- SAFE-T defines a higher-layer packet format carried over the existing
  64-bit S-5701 block framing via segmentation [Ref:AARS5701]
  [Ref:Craven]
- Fields (little-endian byte order):
  - **16-bit length**: total packet byte count; written first in
    UART-framed transmissions
  - **32-bit session ID**: fresh random value generated by HOT per
    pairing attempt; all messages carry this ID, enabling rejection of
    stale or concurrent sessions
  - **8-bit message type**: identifies the message's role in the
    protocol state machine (see Table 1)
  - **variable payload**: content specific to message type
  - **HMAC-SHA-256 truncated to `SIGNATURE_SIZE` bytes**: present on all
    post-pairing authenticated messages; absent on unauthenticated
    pairing messages (advertisements, key exchanges, nonce reveals)
- Message types (direction, authenticated): `HOT_MSG_ADV`
  (HOT→EOT, no), `EOT_MSG_PUBKEY` (EOT→HOT, no),
  `HOT_MSG_PUBKEY_AND_COMMIT` (HOT→EOT, no), `EOT_MSG_NONCE`
  (EOT→HOT, no), `HOT_MSG_NONCE` (HOT→EOT, no), `HOT_MSG_STATUS`
  (HOT→EOT, yes), `EOT_MSG_STATUS` (EOT→HOT, yes), `HOT_MSG_EMERGENCY`
  (HOT→EOT, yes), `EOT_MSG_EMERGENCY` (EOT→HOT, yes),
  `HOT_MSG_DISCONNECT` (HOT→EOT, yes), `EOT_MSG_UPGRADE` (EOT→HOT, no)
- /Recommended table: message enumeration with symbolic name,
  direction, authentication status, and payload description/
- SAFE-T packets exceeding a single block's 56-bit BCH-protected payload
  capacity are segmented across consecutive blocks using an 8-bit
  sequence number field; BCH/FEC remains at the physical framing level;
  the SAFE-T layer treats the radio channel as a byte stream with
  possible loss [Ref:Craven] [Ref:DEFCON26]
- /Recommended figure: annotated SAFE-T packet byte-layout diagram for a
  post-pairing `HOT_MSG_EMERGENCY`, with field names, widths, and a
  callout indicating HMAC coverage/

#### 2.1.1 Packet-format figure

```text
+--------+-------------+----------+-------------------+--------------+
| Length | Session ID  | Msg Type | Payload           | HMAC        |
| 2 B    | 4 B         | 1 B      | variable          | 6 B         |
+--------+-------------+----------+-------------------+--------------+
   \
    \_ included in packet length; HMAC covers Session ID || Msg Type ||
       Payload, but not the length field
```

#### 2.1.2 Message enumeration table

| Symbol | Direction | Authenticated | Payload summary |
| --- | --- | --- | --- |
| `HOT_MSG_ADV` | HOT → EOT | No | Session advertisement; invites pairing. |
| `EOT_MSG_PUBKEY` | EOT → HOT | No | EOT public key. |
| `HOT_MSG_PUBKEY_AND_COMMIT` | HOT → EOT | No | HOT public key and nonce commitment. |
| `EOT_MSG_NONCE` | EOT → HOT | No | EOT nonce reveal. |
| `HOT_MSG_NONCE` | HOT → EOT | No | HOT nonce reveal. |
| `HOT_MSG_STATUS` | HOT → EOT | Yes | Authenticated status request / poll. |
| `EOT_MSG_STATUS` | EOT → HOT | Yes | Authenticated status response. |
| `HOT_MSG_EMERGENCY` | HOT → EOT | Yes | Authenticated emergency brake command. |
| `EOT_MSG_EMERGENCY` | EOT → HOT | Yes | Acknowledgment or mirrored emergency state. |
| `HOT_MSG_DISCONNECT` | HOT → EOT | Yes | Session teardown and return to idle. |
| `EOT_MSG_UPGRADE` | EOT → HOT | No | Legacy-mode upgrade hint for updated peers. |

### 2.2 Authenticated Message Construction

- HMAC input:
  `session_id (4 B) || msg_type (1 B) || payload  (variable)`; the
  length prefix is excluded
- HMAC key: the 32-byte shared secret derived from the ECDH exchange
- Output truncated to `SIGNATURE_SIZE` bytes and appended to the
  packet
- Verification: receiver recomputes HMAC over the same fields and
  performs a constant-time comparison (`ext_memcmp`); any mismatch
  causes silent discard, logged as an invalid signature event
- Including the session ID in HMAC input binds each authenticated
  message to its session, preventing cross-session replay even from a
  validly captured prior-session packet

### 2.3 Per-Session Message Counter and Replay Protection

- Each post-pairing HOT-to-EOT message carries a monotonically
  increasing 32-bit counter (`msg_ctr_t`) as the first bytes of its
  payload; the EOT maintains a per-session high-water mark:

  ``` c
  if (recv_ctr <= conn.ctr) { break; } /* reject: replayed or
  out-of-order */ conn.ctr = recv_ctr;
  ```

- A captured EB command cannot be replayed within the same session
  because its counter value is at or below the high-water mark; the
  counter resets to zero on each new pairing (new session ID, new shared
  secret)
- **Epoch-based replay protection (design extension, not implemented)**:
  each message would carry a timestamp synchronized at pairing; the EOT
  would reject messages older than a configured window (e.g., 15 s),
  closing the jam-and-replay attack vector; implementation is deferred
  because RTC availability across the installed fleet is unconfirmed
  [Ref:STM32F103Datasheet]; the counter mechanism provides the primary
  in-session replay defense in the interim

### 2.4 PIN Generation and the 5-Digit Constraint

- The pairing PIN is a 5-digit decimal value (0–99999; 100,000 possible
  values) derived from both parties' public keys and nonces
  [Ref:BluetoothSpec] [Ref:RFC7748]

  - \*Note: the published IEEE INNOVARail SAFE-T paper refers to a
    "6-digit PIN" and "1 in 1,000,000" probability; these are errata;
    the `compute_pin` implementation uses `% 100000`, and the HOT
    hardware input accommodates exactly 5 digits; all thesis references
    use 5 digits\*

- Derivation:

  ``` c
  /* Pa = EOT pubkey (64 B uncompressed), Pb = HOT pubkey */ /* Na =
  EOT nonce (32 B), Nb = HOT nonce */ data = Pa || Pb || Na || Nb;
  hash = SHA-256(data); pin = (*(uint32_t *)hash) % 100000; /* 5-digit
  PIN */
  ```

- The 5-digit keyspace is adequate given operational constraints:

  - PIN verification is **online-only**: the attacker must interact with
    a live HOT to test a candidate
  - Incorrect PIN entry aborts pairing, clears all session state, and
    forces a fresh ECDH exchange; the HOT allows 3 attempts before
    returning to `HOT_IDLE`, bounding the attacker's per-session
    success probability at 3/100,000 = 0.003%
  - This matches the security model for Bluetooth Passkey Entry
    [Ref:BluetoothSpec], which uses a comparable keyspace and online-only
    requirement

- The EOT displays the PIN on its 8-character alphanumeric LED; the HOT
  operator enters it via the 5-digit numeric input

- The PIN tells the HOT it has paired with the intended EOT; the EOT has
  no independent radio confirmation of the correct HOT—instead, the
  engineer confirms verbally to the rear worker, who presses TEST a
  second time to transition the EOT to PAIRED, using the same
  out-of-band voice channel as the standard pairing workflow
  [Ref:YoutubeConrail] [Ref:YoutubeHangingEOT]

## 3. Pairing Protocol

- /Recommended figure: MSC or swimlane diagram of the full SAFE-T
  pairing sequence, showing the radio message exchange interleaved with
  out-of-band operator steps; this should be the primary visual for this
  chapter/
- Preconditions: EOT has TEST pressed (advertising mode); HOT has ARM
  pressed (listening mode)
  1.  **HOT advertisement**: HOT transmits `HOT_MSG_ADV` with a fresh
      32-bit session ID at 1 s intervals on the downlink frequency
  2.  **EOT key generation**: on receiving `HOT_MSG_ADV`, EOT generates
      a fresh Curve25519 keypair (Pa, ka) [Ref:Bernstein2006]
      [Ref:RFC7748] and transmits Pa as `EOT_MSG_PUBKEY` tagged with the
      session ID
  3.  **HOT key generation and commitment**: HOT generates its keypair
      (Pb, kb) and nonce Nb, computes C = SHA-256(Nb), and transmits
      (Pb ∥ C) as `HOT_MSG_PUBKEY_AND_COMMIT`
  4.  **EOT nonce transmission**: EOT stores C, generates nonce Na,
      transmits Na as `EOT_MSG_NONCE`
  5.  **HOT nonce reveal and PIN computation**: HOT computes
      `shared_secret` = ECDH(kb, Pa) [Ref:RFC7748] and PIN = g(Pa, Pb,
      Na, Nb); transmits Nb as `HOT_MSG_NONCE`; enters `HOT_WAIT_FOR`<sub>
      PIN</sub>
  6.  **EOT commitment verification and PIN display**: EOT receives Nb,
      verifies SHA-256(Nb) == C; on failure, aborts to `EOT_IDLE`; on
      success, computes `shared_secret` = ECDH(ka, Pb) and same PIN;
      displays 5-digit PIN on LED
  7.  **Operator PIN relay**: rear worker reads PIN from EOT display and
      communicates it to the engineer by voice radio (the existing
      out-of-band channel [Ref:YoutubeConrail] [Ref:YoutubeHangingEOT]);
      engineer enters PIN into HOT
  8.  **HOT PIN verification**: match → HOT transitions to PAIRED; 3
      consecutive failures → abort, clear all session state, return to
      `HOT_IDLE`
  9.  **EOT pairing confirmation**: engineer confirms success to rear
      worker; rear worker presses TEST a second time → EOT transitions to
      PAIRED; single-operator workflows may substitute a linker tool for
      this final physical confirmation, but that tool is outside the core
      protocol and does not change the radio message flow
  10. **Armed operation**: both devices in PAIRED state with shared
      session key; all HOT-to-EOT commands authenticated with HMAC-SHA-
      256 and a monotonically increasing counter; precomputed EB HMAC
      stored for low-latency emergency response
- Multi-EOT tie-breaking: if the HOT receives multiple advertisements
  within a 5-second window, it selects the lowest unit ID; this heuristic
  is deterministic and introduces no security-relevant bias

### 3.1 Single-Operator Pairing: The DPS Linker Tool

- The standard pairing sequence assumes simultaneous presence at both
  ends; the DPS linker tool and other single-operator variants (see
  §1.3.1, System Overview) impose additional constraints on the SAFE-T
  state machine [Ref:VimeoLinkHETETD] [Ref:PatentHOTEmergencyCommands]
  [Ref:PatentRemoteArming]
- In the linker workflow, the operator completes all EOT-side steps
  first, then moves to the cab; by the time the HOT completes PIN
  verification (step 8 above), no one is at the rear to perform the
  second TEST press in step 9
- SAFE-T therefore treats linker-mode as an operational wrapper around
  the same pairing protocol, not as a distinct cryptographic mode. The
  firmware may either:
  - require the second TEST press and keep the protocol unchanged, or
  - allow an explicit linker-mode policy that auto-confirms the EOT after
    successful PIN verification
- The latter is a compatibility concession for single-operator
  deployments; it weakens the human-in-the-loop assurance margin but
  preserves interoperability with the rest of the secure state machine
- All modifications to the linker tool would be firmware-only; no
  hardware changes are required

## 4. State Machines

- /Recommended figure: EOT UML state diagram with states
  `EOT_IDLE`, `EOT_WAIT_ADV`,
  `EOT_KEY_EX`<sub>1</sub>, `EOT_KEY_EX`<sub>2</sub>,
  `EOT_PAIRED`, `EOT_LEGACY`; labeled transitions and
  timeout arcs/
- /Recommended figure: HOT UML state diagram with states
  `HOT_IDLE`, `HOT_ADV`,
  `HOT_KEY_EX`<sub>1</sub>,
  `HOT_WAIT_FOR`<sub>PIN</sub>, `HOT_PAIRED`,
  `HOT_WAIT_FOR`<sub>STATUS</sub>,
  `HOT_WAIT_FOR`<sub>EMERGENCY</sub>, `HOT_LEGACY`,
  `HOT_LEGACY_ARMED`, `HOT_WAIT_FOR_LEGACY_EB_ACK`; labeled transitions/
- **EOT states**:
  - **`EOT_IDLE`**: default; transmits legacy S-5701 status
    broadcast [Ref:CFR405]; waits for TEST press
  - **`EOT_WAIT_ADV`**: listens for `HOT_MSG_ADV`;
    times out (30 s) to `EOT_IDLE` if none received
  - **`EOT_KEY_EX`<sub>1</sub>**: has sent Pa; awaits
    `HOT_MSG_PUBKEY``_AND_COMMIT`; pairing timeout
    active
  - **`EOT_KEY_EX`<sub>2</sub>**: has sent Na; awaits
    `HOT_MSG_NONCE` and performs commitment verification;
    pairing timeout active
  - **`EOT_PAIRED`**: monitors for authenticated HOT commands;
    TEST press disconnects and restarts advertisement cycle; no legacy
    broadcasts while paired
  - **`EOT_LEGACY`**: entered via ~5-second TEST hold; transmits
    legacy status broadcast; ignores all SAFE-T advertisements; TEST
    press exits to `EOT_IDLE`
- **HOT states**:
  - **`HOT_IDLE`**: monitors for legacy TEST messages and
    `EOT_MSG_UPGRADE` requests from updated EOTs
  - **`HOT_ADV`**: transmits `HOT_MSG_ADV` at 1 s
    intervals; times out to `HOT_IDLE` if no
    `EOT_MSG_PUBKEY` received
  - **`HOT_KEY_EX`<sub>1</sub>**: has sent Pb and commitment;
    awaits `EOT_MSG_NONCE`; pairing timeout active
  - **`HOT_WAIT_FOR`<sub>PIN</sub>**: key exchange complete;
    awaits operator PIN entry; 3 failures abort to `HOT_IDLE`
    with full session state clear
  - **`HOT_PAIRED`**: sends authenticated status poll, EB, and
    disconnect commands; disarm (all-zeros unit ID) sends
    `HOT_MSG_DISCONNECT` and returns to `HOT_IDLE`
    [Ref:YoutubeConrail]
  - **`HOT_WAIT_FOR`<sub>STATUS</sub> /
    `HOT_WAIT_FOR`<sub>EMERGENCY</sub>**: awaits EOT
    acknowledgment; retransmits on timer if no response within the
    retransmit interval
  - **`HOT_LEGACY` / `HOT_LEGACY_ARMED`**: legacy mode;
    `HOT_LEGACY_ARMED` is blocked if the target EOT previously
    sent `EOT_MSG_UPGRADE` (downgrade prevention). The HOT provides a
    5-second window for the user to press ARM after receiving legacy status.
  - **`HOT_WAIT_FOR_LEGACY_EB_ACK`**: awaits EOT acknowledgment
    of legacy EB command; retransmits every 10 s until acknowledged.

## 5. Edge Cases and Failure Handling

- **Pairing timeout**: all key exchange states
  (`EOT_KEY_EX`<sub>1</sub>, `EOT_KEY_EX`<sub>2</sub>,
  `HOT_KEY_EX`<sub>1</sub>) have a configurable timeout; expiry
  returns the device to idle and zeros all partial session state,
  preventing partial-pairing limbo caused by RF packet loss
- **Incorrect PIN entry**: treated identically whether from an attacker
  probe or operator transcription error; HOT aborts, clears all session
  state, returns to `HOT_IDLE`; full restart takes ~30–60 s;
  expected to be infrequent given short PIN length and direct visual
  confirmation from EOT display
- **Session teardown and reconnection**: extended loss of radio contact
  (connection watchdog timeout) transitions both devices to idle and
  zeros the ECDH-derived secret from SRAM; reconnection requires a full
  pairing sequence
- **Simultaneous pairing in yard environments**: the HOT's random
  session ID prevents cross-session confusion; an EOT receiving multiple
  advertisements selects the first; the HOT selects the lowest-unit-ID
  EOT among simultaneous responders; both rules are deterministic and
  require no coordination
- **Legacy upgrade path**: an updated EOT in `EOT_LEGACY` that
  receives a pairing request from an updated HOT transmits
  `EOT_MSG_UPGRADE` alongside its legacy broadcast; the HOT
  records the unit ID and refuses subsequent legacy-mode pairings for
  that ID, preventing adversary-induced legacy re-pairing after an
  upgrade

## 6. Backward Compatibility Strategy

- SAFE-T uses layered simultaneous broadcasts: the EOT transmits both
  the legacy S-5701 TEST message and the SAFE-T advertisement; the HOT
  monitors for both [Ref:Craven] [Ref:DEFCON26]
  - A legacy HOT sees only the TEST message and proceeds through the
    unmodified arming workflow; it is unaware that a SAFE-T
    advertisement was also present
  - An updated HOT receiving a TEST-only EOT enters legacy mode
    normally; if that EOT has previously sent
    `EOT_MSG_UPGRADE`, the HOT refuses legacy mode with that
    unit ID and prompts for the SAFE-T path
- Downgrade protection (cross-reference: §2.2, Adversary and Security
  Model, selective jamming downgrade attack): an attacker cannot cause
  two updated devices to silently enter legacy mode via radio messages
  alone; SAFE-T devices only enter legacy mode in response to a physical
  button interaction at the rear of the train, never in response to any
  received packet
- /Recommended table: four-cell backward compatibility matrix
  (updated/legacy HOT × updated/legacy EOT) with columns: pairing path
  \| operator actions required \| security properties provided \| notes/

## 7. Design Rationale and Trade-offs

- **Curve25519 over Craven's 512-bit DH [Ref:Craven]**: 512-bit DH parameters
  are now cryptographically weak; Curve25519 provides 128-bit security
  with better Cortex-M3 performance via Montgomery ladder, a cleaner
  security proof, and public-domain status with no patent encumbrance
  relevant to AAR standards adoption [Ref:Bernstein2006] [Ref:BluetoothSpec]
  [Ref:Ondas]
- **HMAC-SHA-256 over stream/block cipher**: provides integrity and
  authentication in a single primitive; requires no IV management or
  nonce synchronization beyond the existing counter; well-characterized
  on constrained hardware with no additional state requirements
- **PIN-based pairing over pre-provisioned keys**: pre-provisioning
  requires centralized key management infrastructure incompatible with
  dark-territory operations; a PIN-based model with fresh per-session
  ECDH material requires only the existing human operator coordination
  channel already mandated by the pairing workflow [Ref:YoutubeConrail]
  [Ref:CFR405]
- **No encryption of status messages**: encrypting EOT-to-HOT telemetry
  would increase packet size and processing overhead without addressing
  the primary safety risk (unauthorized EB activation); the unit ID and
  brake pressure remain observable in cleartext, consistent with the
  current operational model; confidentiality is not a stated regulatory
  requirement; SAFE-T's security does not depend on the unit ID being
  secret, unlike the legacy protocol
- **Precomputed EB HMAC**: the 1-second actuation deadline of 49 CFR §
  232.405 [Ref:CFR405] creates a hard computational budget on the EB receive
  path; while HMAC-SHA-256 on a short message completes well under 1 ms
  at 72 MHz, precomputation eliminates all non-determinism from the
  critical path and provides margin against interrupt latency and future
  firmware additions sharing CPU cycles with the EB handler

## 8. Formal Security Analysis

### 8.1 ProVerif Model Abstraction

- The SAFE-T protocol is formally modeled in ProVerif under the
  Dolev-Yao adversary model [Ref:DolevYao], in which the attacker has
  full control of the radio channel: intercept, read, delay, drop,
  replay, and inject arbitrary messages
- The model follows the standard ProVerif methodology for symbolic
  protocol analysis: stateful roles are encoded as processes, secrecy and
  correspondence claims are expressed as events and queries, and the
  attacker is assumed to control the network algebraically but not the
  cryptographic primitives [Ref:proverif]
- The modeled attacker can instantiate arbitrary numbers of
  attacker-controlled EOT and HOT processes, enabling full MITM attack
  modeling
- The model defines two insecure radio channels (`eot_to_hot`,
  `hot_to_eot`), a private operator channel (simulating out-of-band PIN
  communication), and a private button channel (simulating the physical
  second TEST press); this channel structure corresponds directly to the
  real-world pairing workflow [Ref:YoutubeConrail]
- The ProVerif model (`verif.pv`) is reproduced in Appendix A and is
  summarized in the repository at `reference_impl/verif.pv`
- Physical aspects outside the model: timing side channels, power
  analysis, physical device compromise, BCH decoding errors, and radio
  propagation; these are outside Dolev-Yao scope [Ref:DolevYao]; the
  modeled attacker is strictly stronger than a physical attacker, so the
  result is conservative

### 8.2 Properties Verified

- **Spoofing resistance**: the attacker cannot cause an EOT to execute
  an EB command without possession of the ECDH-derived shared secret; a
  forged packet without the correct key produces an HMAC verification
  failure with overwhelming probability
- **Replay resistance**: the counter-based mechanism prevents in-session
  replay; cross-session replay is prevented by session ID freshness;
  this is the property captured in the current ProVerif model
- **MITM resistance during pairing**: the commitment scheme prevents
  either party from biasing the PIN after learning the other's nonce;
  the 100,000-value keyspace combined with online-only attack
  requirements and 3-attempt HOT-side rate limiting bounds the per-
  session attacker success probability at 3/100,000 = 0.003%
  [Ref:BluetoothSpec]; verified via injective correspondence between
  `PinVerified` and `PinGenerated` events
- **Session independence**: each session uses independently generated
  ECDH key material; verified via agreement between
  `HOTEstablishedSecret` and `EOTEstablishedSecret` events;
  compromise of one session secret does not affect any other session
- **Post-pairing message authenticity**: any `MessageVerified` event
  implies a prior `MessageSent` with the same payload and counter,
  ruling out forgery and replay under the Dolev-Yao model [Ref:DolevYao]

### 8.3 Modeling Limitations

- The ProVerif model does not include epoch-based timestamp replay
  protection or the rate-limiting mechanism; a complete analysis of the
  timestamp defense would require additional modeling of clock
  synchronization accuracy and window parameters [Ref:STM32F103Datasheet]
- Any epoch-based extension must therefore be parameterized by the RTC
  drift and tolerance of the deployed hardware; this thesis leaves that
  extension for future work because the current prototype does not rely
  on an always-on RTC
- The PIN strength argument relies on the HOT correctly implementing the
  3-attempt limit and clearing all state on failure; any implementation
  deviation weakens the MITM resistance claim
- Physical-layer behavior—BCH decoding errors, radio propagation, and
  embedded state machine timing—is not captured; the formal result
  covers the protocol logic only

## 9. Simulation and Testing Environment

- **Current Simulation Model**
  - Uses POSIX sockets or UART for inter-device communication.
  - Substitutes 450 MHz UHF RF transmissions for testing and protocol validation.
  - Abstraction layers separate core cryptography and state machines from the transport layer.
- **Architectural Targets**
  - **Unix (Development & Testing)**
    - Uses POSIX domain sockets.
    - Standard C library for I/O, timing, and randomness.
    - Python orchestrator enables parallel execution via dynamically generated socket paths.
  - **ARM Cortex-M4 (STM32F4 Bare Metal)**
    - Uses UART0 for EOT-HOT messaging.
    - Uses UART1 for external I/O testing logs.
    - Custom xorshift32 PRNG seeded by SysTick.
    - Tested via QEMU emulator with reproducible random seeds.
- **Testing Orchestration**
  - Automated Python framework (`test_orchestrator.py`).
  - Supports hybrid downgrade protection validation and bandwidth profiling.
  - Real-time logging adjustments to prevent interleaved test outputs.
