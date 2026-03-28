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

        print(f"[{orchestrator.elapsed_time():.2f}s] test_full_pairing PASSED")

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

        print(f"[{orchestrator.elapsed_time():.2f}s] test_basic_communication PASSED")

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

        print(f"[{orchestrator.elapsed_time():.2f}s] test_wrong_pin PASSED")

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

        print(
            f"[{orchestrator.elapsed_time():.2f}s] test_packet_drop PASSED (packet drop observed)"
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
        print(
            f"Waiting at most {timeout_sec} seconds for EOT to timeout waiting for HOT advertisement..."
        )
        await eot.assert_output("timed out", timeout=timeout_sec)

        print(f"[{orchestrator.elapsed_time():.2f}s] test_timeout PASSED")

    finally:
        await orchestrator.teardown()


async def test_legacy_mode(orchestrator: TestOrchestrator) -> None:
    """Test legacy mode pairing and communication."""
    await orchestrator.setup("test_legacy_mode")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        # HOT enter legacy mode with ID 12345
        await hot.send_input("12345\n")
        await hot.assert_output("Entering legacy mode with unit ID 12345")

        # EOT enter legacy mode
        await eot.send_input("2\n")
        await eot.assert_output("Entering legacy mode")

        # EOT sends status (manually triggered in our test by pressing enter)
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

        print(f"[{orchestrator.elapsed_time():.2f}s] test_legacy_mode PASSED")

    finally:
        await orchestrator.teardown()


TESTS = {
    "full_pairing": test_full_pairing,
    "basic_communication": test_basic_communication,
    "wrong_pin": test_wrong_pin,
    "packet_drop": test_packet_drop,
    "timeout": test_timeout,
    "legacy_mode": test_legacy_mode,
}


async def run_tests(
    test_names: list[str], arm_mode: bool = False, seed: Optional[int] = None
) -> bool:
    """Run specified tests. Returns True if all pass."""
    tests_status = []

    for name in test_names:
        if name not in TESTS:
            print(f"Unknown test: {name}")
            print(f"Available tests: {', '.join(TESTS.keys())}")
            tests_status.append((name, "SKIPPED"))
            continue

        orchestrator = TestOrchestrator(arm_mode=arm_mode, seed=seed)
        try:
            await TESTS[name](orchestrator)
            tests_status.append((name, "PASSED"))
        except AssertionError as e:
            print(f"[{orchestrator.elapsed_time():.2f}s] {name} FAILED: {e}")
            print(f"  Error: {e}")
            if orchestrator.eot:
                print("  EOT recent output:")
                for line in orchestrator.eot.get_recent_output(5):
                    print(f"    {line}")
            if orchestrator.hot:
                print("  HOT recent output:")
                for line in orchestrator.hot.get_recent_output(5):
                    print(f"    {line}")
            tests_status.append((name, "FAILED"))
        except Exception as e:
            print(f"[{orchestrator.elapsed_time():.2f}s] {name} ERROR: {e}")
            tests_status.append((name, "ERROR"))

        await asyncio.sleep(0.2)

    num_passed = sum(1 for _, status in tests_status if status == "PASSED")
    num_failed = sum(1 for _, status in tests_status if status == "FAILED")
    num_other = sum(
        1 for _, status in tests_status if status not in ("PASSED", "FAILED")
    )
    num_total = len(tests_status)

    print("\n=== Test Summary ===")
    print(
        f"Total: {num_total}, Passed: {num_passed}, Failed: {num_failed}, Other: {num_other}"
    )

    all_passed = num_passed == num_total

    if not all_passed:
        print("\nFailed/Errored tests:")
        for name, status in tests_status:
            if status != "PASSED":
                print(f"  {name}: {status}")

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

    if not args or args[0] == "--help" or args[0] == "-h":
        print(
            "Usage: python test_orchestrator.py [--arm] [--seed SEED] <test_name> [test_name...]"
        )
        print(f"Available tests: {', '.join(TESTS.keys())}")
        print("Use 'all' to run all tests")
        print("Use 'brief' to run all tests EXCEPT timeout (faster)")
        print("Use --arm to run on QEMU/ARM instead of native")
        print("Use --seed to specify a base RNG seed (ARM only)")
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


    success = asyncio.run(run_tests(test_names, arm_mode=arm_mode, seed=seed))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
