#!/usr/bin/env python3.13
"""
Test orchestrator for EOT/HOT device protocol testing.

Runs ./eot and ./hot in parallel, allowing:
- Assertion of output patterns
- Input injection to stdin
- Logging to file for debugging
- ARM/QEMU testing with UART socket bridging
"""

import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from test_utils import TestOrchestrator


async def test_full_pairing(orchestrator: TestOrchestrator) -> None:
    """Test complete pairing flow from idle to paired state."""
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

        orchestrator.log(f"[{orchestrator.elapsed_time():.2f}s] test_full_pairing completed successfully")

    finally:
        await orchestrator.teardown()


async def test_basic_communication(orchestrator: TestOrchestrator) -> None:
    """Test pairing + status request + emergency brake."""
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
        await eot.assert_output("sent status update to HOT", timeout=1)
        await hot.assert_output("EOT Status:")

        await hot.assert_output("Select an option")

        await hot.send_input("2\n")
        await hot.assert_output("sent emergency brake request")
        await eot.assert_output("Emergency brake activated")
        await hot.assert_output("received emergency brake confirmation")

        orchestrator.log(f"[{orchestrator.elapsed_time():.2f}s] test_basic_communication completed successfully")

    finally:
        await orchestrator.teardown()


async def test_wrong_pin(orchestrator: TestOrchestrator) -> None:
    """Test that wrong PIN entry fails and returns to idle."""
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

        orchestrator.log(f"[{orchestrator.elapsed_time():.2f}s] test_wrong_pin completed successfully")

    finally:
        await orchestrator.teardown()


async def test_packet_drop(orchestrator: TestOrchestrator) -> None:
    """Test that packet drops cause timeout and recovery."""
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
            f"[{orchestrator.elapsed_time():.2f}s] test_packet_drop completed successfully (packet drop observed)"
        )

    finally:
        await orchestrator.teardown()


async def test_timeout(orchestrator: TestOrchestrator) -> None:
    """Test that pairing timeout returns devices to idle."""
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

        orchestrator.log(f"[{orchestrator.elapsed_time():.2f}s] test_timeout completed successfully")

    finally:
        await orchestrator.teardown()


async def test_legacy_mode(orchestrator: TestOrchestrator) -> None:
    """Test legacy mode pairing and communication (BOTH devices in legacy_only mode)."""
    await orchestrator.setup("test_legacy_mode", eot_mode="legacy_only", hot_mode="legacy_only")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        # Both devices start in legacy mode automatically
        await eot.assert_output("Legacy-only mode: Entering legacy mode")
        await hot.assert_output("Legacy-only mode: waiting for legacy ARM command")

        # HOT enter unit ID to pair with (legacy only mode requires specifying target ID)
        await hot.send_input("12345\n")
        await hot.assert_output("Legacy only mode active for unit ID 12345")

        # Since EOT already sent ARM command on startup, HOT should already be in ARMED mode or about to be.
        # But wait, EOT sends ARM command on transition to EOT_LEGACY.
        # Let's ensure EOT sends status
        await eot.send_input("\n")
        await eot.assert_output("TEST button pressed, sending legacy status update")

        # HOT receives status and prompts for ARM NOW
        await hot.assert_output("Received status from EOT 12345")

        # HOT ARM NOW
        await hot.send_input("\n")
        await hot.assert_output("SYSTEM ARMED")

        # HOT request legacy status
        await hot.send_input("1\n")
        await hot.assert_output("Requesting legacy status update")
        await eot.assert_output("Received legacy status request")
        await hot.assert_output("EOT Status:")

        # HOT legacy EB
        await hot.send_input("2\n")
        await hot.assert_output("Sending legacy emergency brake request")
        await eot.assert_output("Received legacy emergency brake request")
        await eot.assert_output("Emergency brake activated")
        await hot.assert_output("Received legacy emergency brake acknowledgment")

        orchestrator.log(f"[{orchestrator.elapsed_time():.2f}s] test_legacy_mode completed successfully")

    finally:
        await orchestrator.teardown()


async def test_downgrade_protection(orchestrator: TestOrchestrator) -> None:
    """Test that modern HOT ignores legacy ARM from legacy EOT."""
    await orchestrator.setup("test_downgrade_protection", eot_mode="legacy_only", hot_mode="default")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("Legacy-only mode: Entering legacy mode")
        await hot.assert_output("HOT_IDLE")

        # EOT should send ARM command.
        # HOT should ignore it and log it as a downgrade attempt.
        await hot.assert_output("ignoring legacy ARM command from unit ID 12345 to prevent downgrade attack")

        orchestrator.log(f"[{orchestrator.elapsed_time():.2f}s] test_downgrade_protection completed successfully (HOT correctly ignored legacy ARM)")

    finally:
        await orchestrator.teardown()


async def test_mixed_modes(orchestrator: TestOrchestrator) -> None:
    """Test that pairing fails when only one side is in legacy_only mode."""
    await orchestrator.setup("test_mixed_modes", eot_mode="default", hot_mode="legacy_only")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("Legacy-only mode: waiting for legacy ARM command")

        # EOT tries to do modern pairing
        await eot.send_input("1\n")
        await eot.assert_output("waiting for HOT advertisement")

        # HOT is in legacy mode, it won't send advertisements
        # We wait to make sure HOT doesn't respond to EOT's existence in a way that allows pairing
        await asyncio.sleep(2)
        orchestrator.log("Ensuring HOT remains in legacy mode waiting for ARM")
        # Ensure it didn't transition to any paired state
        assert "EOT Status" not in str(hot.get_recent_output())

        orchestrator.log(f"[{orchestrator.elapsed_time():.2f}s] test_mixed_modes completed successfully (Devices failed to pair as expected)")

    finally:
        await orchestrator.teardown()


TESTS = {
    "full_pairing": test_full_pairing,
    "basic_communication": test_basic_communication,
    "wrong_pin": test_wrong_pin,
    "packet_drop": test_packet_drop,
    "timeout": test_timeout,
    "legacy_mode": test_legacy_mode,
    "downgrade_protection": test_downgrade_protection,
    "mixed_modes": test_mixed_modes,
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
            return name, "SKIPPED", 0.0

        async with semaphore:
            orchestrator = TestOrchestrator(arm_mode=arm_mode, seed=test_seed, baud_rate=baud)
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
    arm_mode = False
    args = sys.argv[1:]

    if "--arm" in args:
        arm_mode = True
        args.remove("--arm")

    seed = None
    if "--seed" in args:
        idx = args.index("--seed")
        if idx + 1 < len(args):
            try:
                seed = int(args[idx + 1])
                args.pop(idx + 1)
                args.pop(idx)
            except ValueError:
                print(f"Error: Invalid seed value '{args[idx+1]}'")
                sys.exit(1)
        else:
            print("Error: --seed requires an integer value")
            sys.exit(1)

    baud = None
    if "--baud" in args:
        idx = args.index("--baud")
        if idx + 1 < len(args):
            try:
                baud = int(args[idx + 1])
                args.pop(idx + 1)
                args.pop(idx)
            except ValueError:
                print(f"Error: Invalid baud value '{args[idx+1]}'")
                sys.exit(1)
        else:
            print("Error: --baud requires an integer value")
            sys.exit(1)

    parallel = False
    if "--parallel" in args:
        parallel = True
        args.remove("--parallel")

    jobs = 4
    if "-j" in args or "--jobs" in args:
        parallel = True
        idx = args.index("-j") if "-j" in args else args.index("--jobs")
        if idx + 1 < len(args):
            try:
                jobs = int(args[idx + 1])
                args.pop(idx + 1)
                args.pop(idx)
            except ValueError:
                # If next arg is not an int, it might be a test name
                args.pop(idx)
        else:
            args.pop(idx)

    if not args or args[0] == "--help" or args[0] == "-h":
        print(
            "Usage: python test_orchestrator.py [--arm] [--seed SEED] [--baud BAUD] [--parallel] [-j JOBS] <test_name> [test_name...]"
        )
        print(f"Available tests: {', '.join(TESTS.keys())}")
        print("Use 'all' to run all tests")
        print("Use 'brief' to run all tests EXCEPT timeout (faster)")
        print("Use --arm to run on QEMU/ARM instead of native")
        print("Use --seed to specify a base RNG seed (ARM only)")
        print("Use --baud to specify a baud rate for the UART bridge (ARM only)")
        print("Use --parallel or -j [N] to run tests in parallel (default jobs=4)")
        sys.exit(1)

    test_names = args
    if "all" in test_names:
        test_names = list(TESTS.keys())
    elif "brief" in test_names:
        test_names = [name for name in TESTS.keys() if name != "timeout"]

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
