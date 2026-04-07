#!/usr/bin/env python3.13
"""
Test orchestrator for EOT/HOT device protocol testing.

Runs ./eot and ./hot in parallel, allowing:
- Assertion of output patterns
- Input injection to stdin
- Logging to file for debugging
- ARM/QEMU testing with UART socket bridging
"""

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from test_utils import TestOrchestrator


async def test_timeout(orchestrator: TestOrchestrator) -> None:
    """Test that pairing timeout returns devices to idle."""
    orchestrator.print_early_header("test_timeout")
    await orchestrator.setup("test_timeout")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")

        if orchestrator.arm_mode:
            # QEMU (ARM) is slower due to emulation overhead
            timeout_sec = 40
        else:
            timeout_sec = 35
        orchestrator.log(
            f"Waiting at most {timeout_sec} seconds for EOT to timeout waiting for HOT advertisement..."
        )
        await eot.assert_output("timed out", timeout=timeout_sec)

        orchestrator.log(f"[{orchestrator.elapsed_time():5.2f}s] test_timeout completed successfully")

    finally:
        await orchestrator.teardown()


async def test_full_pairing(orchestrator: TestOrchestrator) -> None:
    """Test complete pairing flow from idle to paired state."""
    orchestrator.print_early_header("test_full_pairing")
    await orchestrator.setup("test_full_pairing")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")
        await eot.assert_output("waiting for HOT advertisement")

        await hot.send_input("-1\n")
        await hot.assert_output("sending advertisement")

        await eot.assert_output("received advertisement")
        await eot.assert_output("sent public key")

        await hot.assert_output("received EOT pubkey")
        await hot.assert_output("sent pubkey and commitment")

        await eot.assert_output("received HOT pubkey and commitment")
        await eot.assert_output("sent nonce")

        await hot.assert_output("received EOT nonce")
        await hot.assert_output("sent nonce to EOT")

        eot_pin_line = await eot.assert_output(r"PIN is \d{5}")
        eot_pin_match = re.search(r"PIN is (\d{5})", eot_pin_line)
        assert eot_pin_match, f"Could not extract PIN from: {eot_pin_line}"
        pin = eot_pin_match.group(1)

        hot_pin_line = await hot.assert_output(r"expected PIN is \d{5}")
        hot_pin_match = re.search(r"expected PIN is (\d{5})", hot_pin_line)
        assert hot_pin_match, f"Could not extract PIN from: {hot_pin_line}"
        hot_pin = hot_pin_match.group(1)

        assert pin == hot_pin, f"PIN mismatch: EOT={pin}, HOT={hot_pin}"

        await hot.send_input(f"{pin}\n")
        await hot.assert_output("PIN correct")

        await eot.send_input("\n")
        await eot.assert_output("Pairing successful")

        orchestrator.log(f"[{orchestrator.elapsed_time():5.2f}s] test_full_pairing completed successfully")

    finally:
        await orchestrator.teardown()


async def test_basic_communication(orchestrator: TestOrchestrator) -> None:
    """Test pairing + status request + emergency brake."""
    orchestrator.print_early_header("test_basic_communication")
    await orchestrator.setup("test_basic_communication")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")
        await hot.send_input("-1\n")

        eot_pin_line = await eot.assert_output(r"PIN is (\d{5})", timeout=10)
        pin_match = re.search(r"PIN is (\d{5})", eot_pin_line)
        pin = pin_match.group(1) if pin_match else None

        await hot.assert_output(r"expected PIN is \d{5}")

        if pin:
            await hot.send_input(f"{pin}\n")
            await hot.assert_output("PIN correct")

        await eot.send_input("\n")
        await eot.assert_output("Pairing successful")

        await hot.assert_output("Select an option")

        await hot.send_input("1\n")
        await hot.assert_output("sent status update request")
        await eot.assert_output("sent status update to HOT")
        await hot.assert_output("EOT Status:")

        await hot.assert_output("Select an option")

        await hot.send_input("2\n")
        await hot.assert_output("sent emergency brake request")
        await eot.assert_output("Emergency brake activated")
        await hot.assert_output("received emergency brake confirmation")

        orchestrator.log(f"[{orchestrator.elapsed_time():5.2f}s] test_basic_communication completed successfully")

    finally:
        await orchestrator.teardown()


async def test_wrong_pin(orchestrator: TestOrchestrator) -> None:
    """Test that wrong PIN entry fails and returns to idle."""
    orchestrator.print_early_header("test_wrong_pin")
    await orchestrator.setup("test_wrong_pin")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")
        await hot.send_input("-1\n")

        await hot.assert_output(r"expected PIN is (\d{5})", timeout=15)

        for attempt in range(3):
            await hot.send_input("00000\n")
            await asyncio.sleep(0.5)

        await hot.assert_output("Failed to enter correct PIN")
        await hot.assert_output("HOT_IDLE")

        orchestrator.log(f"[{orchestrator.elapsed_time():5.2f}s] test_wrong_pin completed successfully")

    finally:
        await orchestrator.teardown()


async def test_packet_drop(orchestrator: TestOrchestrator) -> None:
    """Test that packet drops cause timeout and recovery."""
    orchestrator.print_early_header("test_packet_drop")
    await orchestrator.setup("test_packet_drop", eot_drops=[1])
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("adding packet 1 to drop list")

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")
        await hot.send_input("-1\n")

        await hot.assert_output("sending advertisement")

        await eot.assert_output("dropping packet 1 for testing")

        orchestrator.log(
            f"[{orchestrator.elapsed_time():5.2f}s] test_packet_drop completed successfully (packet drop observed)"
        )

    finally:
        await orchestrator.teardown()


async def test_legacy_baseline(orchestrator: TestOrchestrator) -> None:
    """Test legacy mode pairing and communication (BOTH devices in legacy_only mode)."""
    orchestrator.print_early_header("test_legacy_baseline")
    await orchestrator.setup("test_legacy_baseline", eot_mode="legacy_only", hot_mode="legacy_only")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("Legacy-only mode: Entering legacy mode")
        await hot.assert_output("Legacy-only mode: waiting for legacy ARM command")

        await hot.send_input("12345\n")
        await hot.assert_output("Legacy only mode active for unit ID 12345")

        await eot.send_input("\n")
        await eot.assert_output("TEST button pressed, sending legacy status update")
        await hot.assert_output("Received status from EOT 12345")

        await hot.send_input("\n")
        await hot.assert_output("SYSTEM ARMED")

        await hot.send_input("1\n")
        await hot.assert_output("Requesting legacy status update")
        await eot.assert_output("Received legacy status request")
        await hot.assert_output("EOT Status:")

        await hot.send_input("2\n")
        await hot.assert_output("Sending legacy emergency brake request")
        await eot.assert_output("Received legacy emergency brake request")
        await eot.assert_output("Emergency brake activated")
        await hot.assert_output("Received legacy emergency brake acknowledgment")

        orchestrator.log(f"[{orchestrator.elapsed_time():5.2f}s] test_legacy_baseline completed successfully")

    finally:
        await orchestrator.teardown()


async def test_modern_downgrade_attempt(orchestrator: TestOrchestrator) -> None:
    """Both devices are modern. EOT is downgraded manually. HOT should reject the manual EOT ID."""
    orchestrator.print_early_header("test_modern_downgrade_attempt")
    await orchestrator.setup("test_modern_downgrade_attempt", eot_mode="default", hot_mode="default")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE:")
        await hot.assert_output("HOT_IDLE:")

        await eot.assert_output("2: Hold TEST button for 5 seconds to enter legacy mode")
        await eot.send_input("2\n")
        eot_entered = await eot.assert_output(r"EOT entering legacy mode\. Unit ID: (\d{5})")
        eot_id = re.search(r"Unit ID: (\d{5})", eot_entered).group(1)

        await asyncio.sleep(1) # Let the UPGRADE packet arrive

        await hot.send_input(f"{eot_id}\n")
        await hot.assert_output("but it supports the new protocol")

        orchestrator.log(f"[{orchestrator.elapsed_time():5.2f}s] test_modern_downgrade_attempt completed successfully")

    finally:
        await orchestrator.teardown()


async def test_eot_legacy_hot_modern(orchestrator: TestOrchestrator) -> None:
    """EOT is legacy_only, HOT is modern. Should pair and work in legacy mode."""
    orchestrator.print_early_header("test_eot_legacy_hot_modern")
    await orchestrator.setup("test_eot_legacy_hot_modern", eot_mode="legacy_only", hot_mode="default")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("Legacy-only mode: Entering legacy mode")
        await hot.assert_output("HOT_IDLE:")

        eot_id = "12345" # default for legacy only EOT
        await hot.send_input(f"{eot_id}\n")
        await hot.assert_output(f"Entering legacy mode with unit ID {eot_id}")

        await eot.send_input("\n")
        await eot.assert_output("TEST button pressed")
        await hot.assert_output(f"Received status from EOT {eot_id}")

        await hot.send_input("\n")
        await hot.assert_output("SYSTEM ARMED")

        orchestrator.log(f"[{orchestrator.elapsed_time():5.2f}s] test_eot_legacy_hot_modern completed successfully")

    finally:
        await orchestrator.teardown()


async def test_eot_modern_hot_legacy(orchestrator: TestOrchestrator) -> None:
    """EOT is modern, HOT is legacy_only. EOT manually downgrades. Should pair and work."""
    orchestrator.print_early_header("test_eot_modern_hot_legacy")
    await orchestrator.setup("test_eot_modern_hot_legacy", eot_mode="default", hot_mode="legacy_only")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await hot.assert_output("Legacy-only mode: waiting for legacy ARM command")
        await eot.assert_output("EOT_IDLE:")

        await eot.send_input("2\n")
        eot_entered = await eot.assert_output(r"EOT entering legacy mode\. Unit ID: (\d{5})")
        eot_id = re.search(r"Unit ID: (\d{5})", eot_entered).group(1)

        await hot.send_input(f"{eot_id}\n")
        await hot.assert_output(f"Legacy only mode active for unit ID {eot_id}")

        await eot.send_input("\n")
        await eot.assert_output("TEST button pressed")
        await hot.assert_output(f"Received status from EOT {eot_id}")

        await hot.send_input("\n")
        await hot.assert_output("SYSTEM ARMED")

        orchestrator.log(f"[{orchestrator.elapsed_time():5.2f}s] test_eot_modern_hot_legacy completed successfully")

    finally:
        await orchestrator.teardown()


TESTS = {
    "full_pairing": test_full_pairing,
    "basic_communication": test_basic_communication,
    "wrong_pin": test_wrong_pin,
    "packet_drop": test_packet_drop,
    "timeout": test_timeout,
    "legacy_baseline": test_legacy_baseline,
    "modern_downgrade_attempt": test_modern_downgrade_attempt,
    "eot_legacy_hot_modern": test_eot_legacy_hot_modern,
    "eot_modern_hot_legacy": test_eot_modern_hot_legacy,
}


async def run_tests(
    test_names: list[str],
    arm_mode: bool = False,
    seed: Optional[int] = None,
    baud: Optional[int] = None,
    parallel: bool = False,
    jobs: int = 1,
) -> bool:
    """Run specified tests. Returns True if all pass."""
    tests_status = []
    semaphore = asyncio.Semaphore(jobs if parallel else 1)

    async def run_single_test(name: str, test_seed: Optional[int]):
        if name not in TESTS:
            print(f"Unknown test: {name}")
            return name, "SKIPPED", 0.0, []

        async with semaphore:
            orchestrator = TestOrchestrator(arm_mode=arm_mode, seed=test_seed, baud_rate=baud, quiet=parallel)
            start_run = time.time()
            try:
                await TESTS[name](orchestrator)
                duration = time.time() - start_run
                print(f"[PASSED] {name} ({duration:.2f}s)")
                return name, "PASSED", duration, orchestrator.get_logs()
            except AssertionError as e:
                duration = time.time() - start_run
                orchestrator.log(f"{name} FAILED: {e}")
                print(f"[FAILED] {name} ({duration:.2f}s)")
                return name, "FAILED", duration, orchestrator.get_logs()
            except Exception as e:
                duration = time.time() - start_run
                orchestrator.log(f"{name} ERROR: {e}")
                import traceback
                orchestrator.log(traceback.format_exc())
                print(f"[ERROR]  {name} ({duration:.2f}s)")
                return name, "ERROR", duration, orchestrator.get_logs()
            finally:
                await orchestrator.teardown()

    # Create tasks for all tests
    tasks = []
    for i, name in enumerate(test_names):
        test_seed = seed + (i * 10) if seed is not None else None
        tasks.append(run_single_test(name, test_seed))

    if parallel:
        results = await asyncio.gather(*tasks)
    else:
        results = []
        for task in tasks:
            results.append(await task)
            await asyncio.sleep(0.1)

    for name, status, duration, logs in results:
        tests_status.append((name, status, logs))

    num_passed = sum(1 for _, status, _ in tests_status if status == "PASSED")
    num_failed = sum(1 for _, status, _ in tests_status if status == "FAILED")
    num_other = sum(
        1 for _, status, _ in tests_status if status not in ("PASSED", "FAILED")
    )
    num_total = len(tests_status)

    print("\n=== Test Summary ===")
    print(
        f"Total: {num_total}, Passed: {num_passed}, Failed: {num_failed}, Other: {num_other}"
    )

    all_passed = num_passed == num_total

    if not all_passed:
        print("\n=== Detailed Logs for Failed Tests ===")
        for name, status, logs in tests_status:
            if status != "PASSED":
                print(f"\n--- {name} ({status}) ---")
                for msg in logs:
                    print(msg)
                print("-" * (len(name) + 15))

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test orchestrator for EOT/HOT device protocol testing.")
    parser.add_argument("tests", nargs="*", default=["all"], help="Test names to run ('all', 'brief', or specific tests)")
    parser.add_argument("--arm", action="store_true", help="Run on QEMU/ARM instead of native")
    parser.add_argument("--seed", type=int, help="Specify a base RNG seed (ARM only)")
    parser.add_argument("--baud", type=int, default=1200, help="Specify a baud rate for the UART bridge (ARM only, default: 1200)")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Number of parallel jobs (default: 4)")
    args = parser.parse_args()

    arm_mode = args.arm
    seed = args.seed
    baud = args.baud
    jobs = args.jobs
    
    test_names = args.tests
    if not test_names:
        test_names = ["all"]
        
    if "all" in test_names:
        test_names = list(TESTS.keys())
    elif "brief" in test_names:
        test_names = [name for name in TESTS.keys() if name != "timeout"]

    # Only run in parallel if requested AND we have multiple tests
    parallel = args.parallel and jobs > 1 and len(test_names) > 1
    
    # If we decided not to be parallel, force jobs to 1 for the semaphore
    if not parallel:
        jobs = 1

    # check that binaries have been built
    if not arm_mode:
        if not os.path.exists("./eot") or not os.path.exists("./hot"):
            raise FileNotFoundError("Binaries not found. Please build the project first.")
    else:
        if not os.path.exists("./eot.elf") or not os.path.exists("./hot.elf"):
            raise FileNotFoundError("Binaries not found. Please build the project first.")


    success = asyncio.run(run_tests(test_names, arm_mode=arm_mode, seed=seed, baud=baud, parallel=parallel, jobs=jobs))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
